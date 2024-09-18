from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
import math
from mamba_block import MambaDecoder
from mamba_config import MambaConfig
import os, json

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1, 
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MambaModel(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        max_sequence_length: int
    ) -> None:
        super().__init__()

        self.config: MambaConfig = config
        self.max_sequence_length = max_sequence_length
        self.fp16_lm_cross_entropy = False
        self.parallel_output = True
        self.share_embeddings_and_output_weights = True
        
        
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

        self.decoder = MambaDecoder(
            config = self.config
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=self.config.num_layers,
            )
        )

    def forward(
        self,
        input_ids
    ) -> Tensor:
            
        decoder_input = self.embedding(input_ids)
            
        decoder_input = decoder_input.permute(1,0,2)
            
        hidden_states = self.decoder(
            hidden_states=decoder_input
        )
        
        logits = hidden_states @ self.embedding.weight.T
        return logits.contiguous()

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        json_config = json.load(open(os.path.join(model_name, "config.json")))
        #state_dict = load_state_dict_hf(model_name)
        
        config = MambaConfig(
        num_layers = json_config["num_hidden_layers"],
        hidden_size = json_config["hidden_size"],
        state_size = json_config["state_size"],
        conv_dimension = json_config["conv_dimension"],
        expansion_factor = json_config["expansion_factor"],
        rms_norm = True,
        use_mem_mlp = True,
        num_attention_heads = json_config["num_attention_heads"],
        num_mem_heads = json_config["num_attention_heads"],
        mamba_headdim = json_config["mamba_headdim"],
        layer_mapping = json_config["layers_block_type"],
        add_bias_linear = json_config["add_bias_linear"],
        use_shared_block_lora = json_config["use_shared_block_lora"],
        lora_rank = json_config["lora_rank"],
        gated_linear_unit = json_config["gated_linear_unit"],
        kv_channels = json_config["kv_channels"],
        ffn_hidden_size = json_config["ffn_hidden_size"],
        vocab_size = json_config["vocab_size"],
        num_mem_blocks = json_config["num_mem_blocks"],
        )
        model = MambaModel(config = config, max_sequence_length = 4096)
        #model.load_state_dict(state_dict)
        return model
