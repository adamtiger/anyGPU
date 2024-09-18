from mamba_config import MambaConfig
from mamba_layer import MambaLayer
from rotary import RotaryEmbedding
from attention import CausalSelfAttention

from contextlib import nullcontext
from functools import partial
import torch.nn as nn
import torch


# these were calculated from triton, but none is handled too
RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None  



# decoder related submodules

class Memory_AttentionBlock(nn.Module):
    def __init__(
        self, config, mixer_cls, norm_cls=nn.LayerNorm, residual_in_fp32=False, fused_add_norm=False
    ):
        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls(config)
        assert config.rms_norm, 'Memory_AttentionBlock only supports RMSNorm'
        self.norm = norm_cls(2 * config.hidden_size)
        self.fused_add_norm = fused_add_norm


    def forward(
        self, hidden_states, residual = None, inference_params=None, attention_mask=None, rotary_pos_emb=None, forward_layer_idx = None
    ):
        
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        
        hidden_states = hidden_states.transpose(0,1).contiguous()
        
        hidden_states = self.mixer(hidden_states, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask, inference_params=inference_params, forward_layer_idx = forward_layer_idx)
        
        hidden_states = hidden_states.transpose(0,1)
        
        return hidden_states


class MambaBlock(nn.Module):
    def __init__(
        self, config, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):

        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(config)
        if config.use_module_layernorm and not config.rms_norm:
            self.norm = norm_cls
        else:
            self.norm = norm_cls(config.hidden_size)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            assert config.num_mem_heads == 0, 'args.num_mem_heads > 0 only supports fused_add_norm=False'
        self.moe = None

    def forward(
        self, hidden_states, from_shared_proj = None, from_tf = None, residual = None
    ):
        
        if not self.fused_add_norm:
            
            residual = (hidden_states + residual) if residual is not None else hidden_states
            if from_tf is not None:
                hidden_states = self.norm((residual + from_tf).to(dtype=self.norm.weight.dtype))
            else:
                hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
        
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
            
        hidden_states = self.mixer(hidden_states, from_shared_proj=from_shared_proj)
        
        return hidden_states , residual


class vBlock(nn.Module):
    def __init__(
        self, config, sa_cls, norm_cls=nn.LayerNorm
    ):
        super().__init__()
        self.use_mem_mlp = config.use_mem_mlp
        self.sa = Memory_AttentionBlock(config, mixer_cls=sa_cls, norm_cls=norm_cls, residual_in_fp32=config.residual_in_fp32)

    def forward(self, hidden_states, residual=None, x_orig=None, inference_params=None, attention_mask=None, rotary_pos_emb=None, forward_layer_idx = None):
        x = hidden_states + residual if residual is not None else hidden_states
        x_ = torch.concatenate([x, x_orig], dim=-1).type(hidden_states.dtype)
        x = self.sa(x_, inference_params=inference_params, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb, forward_layer_idx = forward_layer_idx)
        return x


def count_mem_blocks_in_config(config):
    num_gs = 0
    for val in config.layer_mapping:
        if val == 'g':
            num_gs +=1
    return num_gs

def create_block(config, layer_idx):
    factory_kwargs = {}
    
    if layer_idx == -1:
        num_gs = count_mem_blocks_in_config(config)
        norm_cls = partial(RMSNorm, eps=config.layernorm_epsilon, dtype=torch.float32)
        sa_cls = partial(CausalSelfAttention, **factory_kwargs, layer_number=-1, num_mem_blocks=num_gs)
        block = vBlock(
            config,
            sa_cls=sa_cls,
            norm_cls=norm_cls,
        )
    else: 
        norm_cls = partial(nn.LayerNorm if not config.rms_norm else RMSNorm, eps=config.layernorm_epsilon)
        
        mixer_cls = partial(MambaLayer, layer_idx=layer_idx, **factory_kwargs)
        block = MambaBlock(
            config,
            mixer_cls=mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
        )
    return block


# decoder implementation

class MambaDecoder(nn.Module):
    def __init__(
        self,
        config: MambaConfig
    ):
        super().__init__()

        self.config: MambaConfig = config

        self.post_layer_norm = True
        self.pre_process = True
        self.post_process = True

        self.input_tensor = None

        self.checkpoint_core_block = self.config.recompute_granularity == 'selective'

        self.num_layers_per_pipeline_rank = (
            self.config.num_layers
        )
        
        self.layer_mapping = config.layer_mapping

        self.use_mem_rope = config.use_mem_rope
        

        self._build_layers()

    def _build_layers(self):
        num_layers_to_build = self.num_layers_per_pipeline_rank
        self.layers = torch.nn.ModuleList([create_block(self.config, i + 1) for i in range(num_layers_to_build)])
        if self.config.num_mem_heads > 0:
            blocks = []
            for _ in range(self.config.num_mem_blocks):
                blocks.append(create_block(self.config, layer_idx=-1))
            self.blocks = torch.nn.ModuleList(blocks)
        
            self.block_map = torch.nn.ModuleList([
                nn.Linear(self.config.hidden_size, self.config.hidden_size, bias = self.config.add_bias_linear) if (i%2 == 1 if (self.layer_mapping is None) else self.layer_mapping[i] == 'g') else nn.Identity() for i in range(self.config.num_layers)]) 
            if self.use_mem_rope:
                self.rotary_pos_emb = RotaryEmbedding(
                        2 * self.config.hidden_size // self.config.num_mem_heads, rotary_percent=1.0, seq_len_interpolation_factor=None
                    )

        if self.config.use_low_rank_mamba_proj:
            blocks = []
            d_inner = self.config.expansion_factor * self.config.hidden_size
            nheads = d_inner // self.config.mamba_headdim
            d_in_proj = 2 * d_inner + 2 * self.config.mamba_ngroups * self.config.state_size + nheads
            for _ in range(self.config.num_shared_mamba_proj):
                blocks.append(nn.Linear(self.config.hidden_size, d_in_proj, bias = self.config.add_bias_linear))
            self.in_projs = torch.nn.ModuleList(blocks)


        if self.post_process and self.post_layer_norm:
            self.final_layernorm = RMSNorm(self.config.hidden_size, eps=self.config.layernorm_epsilon, dtype=torch.float32)

    def forward(self, hidden_states):

        rng_context = nullcontext()
        fp8_context = nullcontext()

        with rng_context and fp8_context:
            residual = None
            x_orig = torch.clone(hidden_states)
            from_tf = None
            block_count = 0
            rotary_pos_emb=None
            if self.use_mem_rope:
                rotary_seq_len = hidden_states.shape[1]
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
            for i, layer in enumerate(self.layers):
                if self.config.num_mem_heads > 0:
                    if (i%2 == 1 if (self.layer_mapping is None) else self.layer_mapping[i] == 'g'):
                        from_tf = self.block_map[i](
                            self.blocks[block_count % self.config.num_mem_blocks](
                                hidden_states, residual, x_orig, rotary_pos_emb=rotary_pos_emb, forward_layer_idx=block_count
                            )
                        )
                        block_count += 1
                    else:
                        from_tf, _ = (None, None)
                from_shared_proj = None
                if self.config.use_low_rank_mamba_proj:
                    from_shared_proj = self.in_projs[i % self.config.num_shared_mamba_proj](hidden_states)
                hidden_states, residual = layer(
                    hidden_states=hidden_states,
                    from_shared_proj=from_shared_proj,
                    from_tf=from_tf,
                    residual = residual
                )

        if self.post_process and self.post_layer_norm:
            if not self.config.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.final_layernorm(residual.to(dtype=self.final_layernorm.weight.dtype))
            else:
                fused_add_norm_fn = rms_norm_fn if isinstance(self.final_layernorm, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.final_layernorm.weight,
                    self.final_layernorm.bias,
                    eps=self.final_layernorm.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )

        return hidden_states
