from sdp_test_cases import * 
from sf_test_cases import *
from ops_test_cases import *
from fused_test_cases import *

if __name__ == '__main__':

    path = r"C:\Data\AI\projects\anyGPU\artifacts"

    # generate_sdp_fwd_nomask_noscore(path, "test_sdp_fwd_nomask_noscore_f32_16_64", 16, 64)
    # generate_sdp_bwd_nomask_noscore(path, "test_sdp_bwd_nomask_noscore_f32_16_64", 16, 64)
    # generate_softmax_bwd(path, "test_softmax_bwd_f32_16_64", 16, 64)
    # generate_sdp_fwd_masked_scaled(path, "test_sdp_fwd_masked_scaled_f32_16_128", 16, 128)

    # generate_sf_file_reading(path, "test_sf_diffuser_tensors")

    # generate_layer_norm_fwd_f32(path, "test_layer_norm_fwd_f32")
    # generate_rms_norm_fwd_f32(path, "test_rms_norm_fwd_f32")
    # generate_silu_fwd_f32(path, "test_silu_fwd_f32")
    # generate_gelu_fwd_f32(path, "test_gelu_fwd_f32")
    # generate_gelu_approx_fwd_f32(path, "test_gelu_approx_fwd_f32")
    # generate_embedding_fwd_f32(path, "test_embedding_fwd_f32")
    # generate_rotary_embedding_fwd_f32(path, "test_rotary_embedding_fwd_f32")
    # generate_alt_rotary_embedding_fwd_f32(path, "test_alt_rotary_embedding_fwd_f32")
    # generate_linear_fwd_f32(path, "test_linear_fwd_f32")
    # generate_transpose_fwd_f32(path, "test_transpose_fwd_f32")
    # generate_concat_fwd_f32(path, "test_concat_fwd_f32")
    # generate_repeat_fwd_f32(path, "test_repeat_fwd_f32")
    # generate_slice_fwd_f32(path, "test_slice_fwd_f32")

    #generate_causal_conv1d_fwd_f32(path, "test_causal_conv1d_fwd_f32")

    from dnninspect.tensor import load_tensor
    from dnninspect.tensor import save_tensor
    import torch

    attn_mask = load_tensor(r"C:\Data\AI\projects\anyGPU\artifacts\xgemma2_tests\test_gemma2model_decoder\in_attention_mask.dat")

    min_dtype = torch.finfo(torch.float32).min
    sliding_window_mask = torch.tril(
        torch.ones_like(attn_mask, dtype=torch.bool), diagonal=-4096
    )
    attention_mask = torch.where(sliding_window_mask, min_dtype, attn_mask)

    save_tensor(attention_mask, r"C:\Data\AI\projects\anyGPU\artifacts\xgemma2_tests\test_gemma2model_decoder\in_attention_mask_sliced.dat")
