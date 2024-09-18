import torch.nn as nn
import torch


class RotaryEmbedding(nn.Module):

    def __init__(
        self, kv_channels: int, rotary_percent: float, seq_len_interpolation_factor: float = None
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device())
                / dim
            )
        )

    def forward(self, max_seq_len: int, offset: int = 0):
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.outer(seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb[:, None, None, :]
        return emb


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    rot_dim = freqs.shape[-1]

    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    
    t = (t * cos_) + (_rotate_half(t) * sin_)
    
    return torch.cat((t, t_pass), dim=-1)
