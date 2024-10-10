"""
    In this module the mamba_chunk_scan_combined function
    will be tested against reference implementations.

    The goal is to find the reference implementation of the
    mamba_chunk_scan_combined function.
"""

from dnninspect.tensor import load_tensor
from os.path import join as pjoin
from einops import rearrange, repeat
import torch.nn.functional as F
import torch


"""
   These are notes to help reproduction:
        - Looking for mamba_chunk_scan_combined.

        - In the mamba_ssm python lib there is a function *mamba_split_conv1d_scan_ref*.
        This function contains causal_conv1d and rmsnorm too. They are already implemented
        separately.
        This function uses *ssd_selective_scan* internally. But it is different.
        - Is ssd_chunk_scan_combined_ref has anythin in common with mamba_chunk_scan_combined.

"""

# implementation of the internal logics of chunk scan combined

def chunk_cumsum_ref(dt, A, chunk_size, dt_bias=None, dt_softplus=True):
    """
        dt - (batch, seqlen, nheads)
        A -  (nheads,)
    """
    _, seqlen, _ = dt.shape
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()  # We want high precision for this before cumsum
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    return dA_cumsum, dt


def chunk_state_ref(B, x, dt, dA_cumsum):
    """
    Args:
        B - (batch, seqlen, ngroups, headdim)
        x - (batch, seqlen, nheads, headdim)
        dt - (batch, nheads, nchunks, chunk_size)
        dA_cumsum - (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    states = torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)
    return states


def state_passing_ref(states, dA_chunk_cumsum, initial_states=None):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
        initial_states: (batch, nheads, dim)
    Return:
        out: (batch, nchunks, nheads, dim)
        final_states: (batch, nheads, dim)
    """
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
    dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    # (batch, nheads, nchunks, nchunks)
    dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    # (batch, nheads, nchunks, nchunks)
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
    return out[:, :-1], out[:, -1]


def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    
    Equivalent (this is a huntch):
        CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
        out, _ = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=None, seq_idx=seq_idx)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert C.shape == B.shape
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        C = F.pad(C, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    print(B.shape, C.shape, CB.shape)
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    print(decay.shape)
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out[:, :seqlen, :, :]


# mamba_chunk_scan_combined reference implementation
# simplifications for zamba2:
#  - cu_seqlens == None
#  - z == None
def mamba_chunk_scan_combined_ref(  
        x, dt, A, B, C, chunk_size, 
        D=None, dt_bias=None, 
        initial_states=None, 
        seq_idx=None, 
        dt_softplus=False):
    
    # check input sizes for correctness

    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    dA_cumsum, dt = chunk_cumsum_ref(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)

    states = chunk_state_ref(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)

    states, final_states = state_passing_ref(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None
    )
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    states = states.to(states_dtype)

    out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D)
    return out, final_states


def test_mamba_chunk_scan_combined():
    path = r"C:\Data\AI\projects\anyGPU\artifacts\zamba2_tests\test_mamba2layer_chunk_scan_comb"
    
    x = load_tensor(pjoin(path, "in_0.dat"))
    dt = load_tensor(pjoin(path, "in_1.dat"))
    A = load_tensor(pjoin(path, "in_2.dat"))
    B = load_tensor(pjoin(path, "in_3.dat"))
    C = load_tensor(pjoin(path, "in_4.dat"))
    D = load_tensor(pjoin(path, "in_D.dat"))
    dt_bias = load_tensor(pjoin(path, "in_dt_bias.dat"))
    
    out, fs = mamba_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size=256, D=D, dt_bias=dt_bias, dt_softplus=True)

    exp_out = load_tensor(pjoin(path, "out_0.dat"))
    exp_fs = load_tensor(pjoin(path, "out_1.dat"))

    is_out_correct = not torch.any(
        torch.logical_and(torch.abs(out - exp_out) > 1e-3, torch.abs(out - exp_out) / (torch.abs(out) + torch.abs(exp_out) + 1e-5) * 2.0 > 1e-2)
    )
    is_fs_correct = torch.allclose(fs, exp_fs)  # unstable to calculate fs

    print(f"Output (0): {is_out_correct}")
    print(f"Output (1): {is_fs_correct}")

    print("OUT")
    print(exp_out.flatten()[:10])
    print("Actual:")
    print(out.flatten()[:10])


if __name__ == '__main__':
    test_mamba_chunk_scan_combined()
