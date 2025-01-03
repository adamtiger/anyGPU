"""
    In this module the mamba_chunk_scan_combined function
    will be tested against reference implementations.

    The goal is to find the reference implementation of the
    mamba_chunk_scan_combined function.
"""

from dnninspect.tensor import save_tensor
from dnninspect.tensor import load_tensor
from os.path import join as pjoin
from einops import rearrange, repeat
import torch.nn.functional as F
import torch
import os


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
    if seqlen < nchunks * chunk_size:  # this was not in the original code, this is for fixing their error!
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


def test_mamba_combined_cumsum(path, chunk_size = 256):    
    dt = load_tensor(pjoin(path, "in_0.dat"))
    A = load_tensor(pjoin(path, "in_1.dat"))
    dt_bias = load_tensor(pjoin(path, "in_dt_bias.dat"))
    
    dA_cumsum, dt = chunk_cumsum_ref(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True)

    exp_dA_cumsum = load_tensor(pjoin(path, "out_0.dat"))
    exp_dt = load_tensor(pjoin(path, "out_1.dat"))

    is_out_correct = not torch.any(
        torch.logical_and(torch.abs(dA_cumsum - exp_dA_cumsum) > 1e-3, torch.abs(dA_cumsum - exp_dA_cumsum) / (torch.abs(dA_cumsum) + torch.abs(exp_dA_cumsum) + 1e-5) * 2.0 > 1e-2)
    )
    is_fs_correct = torch.allclose(dt, exp_dt)

    print(f"Output (0): {is_out_correct}")
    print(f"Output (1): {is_fs_correct}")

    print("OUT")
    print(exp_dA_cumsum.flatten()[:10])
    print("Actual:")
    print(dA_cumsum.flatten()[:10])


def test_mamba_combined_chunkstate(path):    
    B = load_tensor(pjoin(path, "in_0.dat"))
    x = load_tensor(pjoin(path, "in_1.dat"))
    dt = load_tensor(pjoin(path, "in_2.dat"))
    dA_cumsum = load_tensor(pjoin(path, "in_3.dat"))
    
    states = chunk_state_ref(B, x, dt, dA_cumsum)

    exp_states = load_tensor(pjoin(path, "out_0.dat"))

    is_out_correct = not torch.any(
        torch.logical_and(1e+2 > torch.abs(exp_states), 
            torch.logical_and(
                torch.abs(exp_states) > -1e-2, torch.abs(states - exp_states) / (torch.abs(states) + torch.abs(exp_states) + 1e-8) * 2.0 > 1e-2)
        )
    )

    checks = torch.logical_and(1e+2 > torch.abs(exp_states), 
        torch.logical_and(
            torch.abs(exp_states) > -1e-2, torch.abs(states - exp_states) / (torch.abs(states) + torch.abs(exp_states) + 1e-8) * 2.0 > 2e-2)
    ).flatten()

    print(f"Output (0): {is_out_correct}")

    print("OUT")
    print(exp_states.flatten()[:10])
    print("Actual:")
    print(states.flatten()[:10])

    print(checks[10000:10100])
    print(torch.abs(states - exp_states).flatten()[10000:10100])
    print(torch.sum(checks) / checks.size(0))


def test_mamba_combined_statepass(path):    
    states = load_tensor(pjoin(path, "in_0.dat"))
    dA_chunk_cumsum = load_tensor(pjoin(path, "in_1.dat"))
    
    states, final_states = state_passing_ref(states, dA_chunk_cumsum, initial_states=None)

    exp_states = load_tensor(pjoin(path, "out_0.dat"))
    exp_fs = load_tensor(pjoin(path, "out_1.dat"))

    is_out_correct = not torch.any(
        torch.logical_and(torch.abs(states - exp_states) > 1e-3, torch.abs(states - exp_states) / (torch.abs(states) + torch.abs(exp_states) + 1e-5) * 2.0 > 1e-2)
    )
    is_fs_correct = torch.allclose(final_states, exp_fs)

    print(f"Output (0): {is_out_correct}")
    print(f"Output (1): {is_fs_correct}")

    print("OUT")
    print(exp_states.flatten()[:10])
    print("Actual:")
    print(states.flatten()[:10])

    print(torch.min(final_states), torch.max(final_states))


def test_mamba_combined_chunkscan(path):    
    B = load_tensor(pjoin(path, "in_0.dat"))
    C = load_tensor(pjoin(path, "in_1.dat"))
    x = load_tensor(pjoin(path, "in_2.dat"))
    dt = load_tensor(pjoin(path, "in_3.dat"))
    dA_cumsum = load_tensor(pjoin(path, "in_4.dat"))
    prev_states = load_tensor(pjoin(path, "in_5.dat"))
    D = load_tensor(pjoin(path, "in_6.dat"))
    
    out = chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=D, z=None)

    exp_out = load_tensor(pjoin(path, "out_0.dat"))

    checks = torch.logical_and(1e+5 > torch.abs(exp_out), 
        torch.logical_and(
            torch.abs(exp_out) > -1e-5, torch.abs(out - exp_out) / (torch.abs(out) + torch.abs(exp_out) + 1e-8) * 2.0 > 5e-2)
    ).flatten()

    print("OUT")
    print(exp_out.flatten()[:10])
    print("Actual:")
    print(out.flatten()[:10])

    print(checks[1000:1100])
    print(torch.abs(out - exp_out).flatten()[1000:1100])
    print(torch.sum(checks) / checks.size(0))


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

    checks = torch.logical_and(1e+5 > torch.abs(exp_out), 
        torch.logical_and(
            torch.abs(exp_out) > -1e-5, torch.abs(out - exp_out) / (torch.abs(out) + torch.abs(exp_out) + 1e-8) * 2.0 > 5e-2)
    ).flatten()

    print("OUT")
    print(exp_out.flatten()[:10])
    print("Actual:")
    print(out.flatten()[:10])

    print(checks[1000:1100])
    print(torch.abs(out - exp_out).flatten()[1000:1100])
    print(torch.sum(checks) / checks.size(0))


def _generate_mamba_chunk_scan_comb_test_case(
    mamba_chunk_fld, 
    batch,
    seqlen,
    nheads, 
    headdim,
    ngroups,
    dstate,
    chunk_size):

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)
    dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32)
    A = torch.randn(nheads, dtype=torch.float32)
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32)
    D = torch.randn(nheads, dtype=torch.float32)
    dt_bias = torch.randn(nheads, dtype=torch.float32)

    # execute the ground truth
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    y, last_state = mamba_chunk_scan_combined(
        x.cuda(),
        dt.cuda(),
        A.cuda(),
        B.cuda(), 
        C.cuda(),
        chunk_size=chunk_size,
        D=D.cuda(),
        z=None,
        dt_bias=dt_bias.cuda(),
        dt_softplus=True,
        seq_idx=None,
        return_final_states=True
    )

    # saving results

    save_tensor(x, pjoin(mamba_chunk_fld, 'x.dat'))
    save_tensor(dt, pjoin(mamba_chunk_fld, 'dt.dat'))
    save_tensor(A, pjoin(mamba_chunk_fld, 'A.dat'))
    save_tensor(B, pjoin(mamba_chunk_fld, 'B.dat'))
    save_tensor(C, pjoin(mamba_chunk_fld, 'C.dat'))
    save_tensor(D, pjoin(mamba_chunk_fld, 'D.dat'))
    save_tensor(dt_bias, pjoin(mamba_chunk_fld, 'dt_bias.dat'))
    save_tensor(y, pjoin(mamba_chunk_fld, 'y.dat'))
    save_tensor(last_state, pjoin(mamba_chunk_fld, 'last_state.dat'))


def generate_mamba_chunk_scan_comb_test_case_1(path: str):

    # parameters, sizes
    batch = 1
    seqlen = 24
    nheads = 64 
    headdim = 128
    ngroups = 1
    dstate = 64

    chunk_size = 256

    mamba_chunk_fld = pjoin(path, "mamba_chunk_sc_1") 
    os.mkdir(mamba_chunk_fld)

    _generate_mamba_chunk_scan_comb_test_case(
        mamba_chunk_fld,
        batch, seqlen, nheads, 
        headdim, ngroups, dstate,
        chunk_size
    )


def generate_mamba_chunk_scan_comb_test_case_2(path: str):

    # parameters, sizes
    batch = 1
    seqlen = 20
    nheads = 64 
    headdim = 64
    ngroups = 2
    dstate = 128

    chunk_size = 128

    mamba_chunk_fld = pjoin(path, "mamba_chunk_sc_2") 
    os.mkdir(mamba_chunk_fld)

    _generate_mamba_chunk_scan_comb_test_case(
        mamba_chunk_fld,
        batch, seqlen, nheads, 
        headdim, ngroups, dstate,
        chunk_size
    )


def test_mamba_chunk_scan_combined_gens(path, chunk_size=256):
    
    x = load_tensor(pjoin(path, "x.dat"))
    dt = load_tensor(pjoin(path, "dt.dat"))
    A = load_tensor(pjoin(path, "A.dat"))
    B = load_tensor(pjoin(path, "B.dat"))
    C = load_tensor(pjoin(path, "C.dat"))
    D = load_tensor(pjoin(path, "D.dat"))
    dt_bias = load_tensor(pjoin(path, "dt_bias.dat"))
    
    out, fs = mamba_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size=chunk_size, D=D, dt_bias=dt_bias, dt_softplus=True)

    exp_out = load_tensor(pjoin(path, "y.dat"))
    exp_fs = load_tensor(pjoin(path, "last_state.dat"))

    is_out_correct = not torch.any(
        torch.logical_and(torch.abs(out - exp_out) > 1e-3, torch.abs(out - exp_out) / (torch.abs(out) + torch.abs(exp_out) + 1e-5) * 2.0 > 1e-2)
    )
    is_fs_correct = torch.allclose(fs, exp_fs)  # unstable to calculate fs

    print(f"Output (0): {is_out_correct}")
    print(f"Output (1): {is_fs_correct}")

    print("OUT")
    print(exp_out.flatten()[:-10])
    print("Actual:")
    print(out.flatten()[:-10])


if __name__ == '__main__':
    #test_mamba_chunk_scan_combined()
    #generate_mamba_chunk_scan_comb_test_case_1("/home/ubuntu/zamba2_inspect")
    #generate_mamba_chunk_scan_comb_test_case_2("/home/ubuntu/zamba2_inspect")
    #test_mamba_chunk_scan_combined_gens(r"C:\Data\AI\projects\anyGPU\artifacts\zamba2_tests\zamba2_inspect\mamba_chunk_sc_1", chunk_size=256)
    #test_mamba_chunk_scan_combined_gens(r"C:\Data\AI\projects\anyGPU\artifacts\zamba2_tests\zamba2_inspect\mamba_chunk_sc_2", chunk_size=128)
    # test_mamba_combined_cumsum(
    #     r"C:\Data\AI\projects\anyGPU\artifacts\zamba2_tests\zamba2_inspect\mamba_combined_cumsum_2", 
    #     chunk_size = 128)
    
    test_mamba_combined_chunkstate(r"C:\Data\AI\projects\anyGPU\artifacts\zamba2_tests\zamba2_inspect\mamba_combined_chunkstate_1")

    #test_mamba_combined_statepass(r"C:\Data\AI\projects\anyGPU\artifacts\zamba2_tests\zamba2_inspect\mamba_combined_statepass_1")

    #test_mamba_combined_chunkscan(r"C:\Data\AI\projects\anyGPU\artifacts\zamba2_tests\zamba2_inspect\mamba_combined_chunkscan_2")
