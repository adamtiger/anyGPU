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
    import torch

    attn_mask = load_tensor(r"C:\Data\AI\projects\anyGPU\artifacts\xgemma2_tests\gemma2_inspect\gemma2model_update_mask\in_0.dat")
    input_tensor = load_tensor(r"C:\Data\AI\projects\anyGPU\artifacts\xgemma2_tests\gemma2_inspect\gemma2model_update_mask\in_1.dat")
    cache_pos = load_tensor(r"C:\Data\AI\projects\anyGPU\artifacts\xgemma2_tests\gemma2_inspect\gemma2model_update_mask\in_2.dat")

    trg_len = 41

    def update_mask(attn_mask, input_tensor, cache_pos, trg_len):
        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = trg_len

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        
        temp_mask = torch.arange(target_length) > cache_pos.reshape(-1, 1)
        causal_mask *= temp_mask
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attn_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attn_mask[:, None, None, :]
        print(padding_mask)
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )
        return causal_mask

    # execute
    causal_mask = update_mask(attn_mask, input_tensor, cache_pos, trg_len)

    out = load_tensor(r"C:\Data\AI\projects\anyGPU\artifacts\xgemma2_tests\gemma2_inspect\gemma2model_update_mask\out_0.dat")

    print(torch.allclose(causal_mask, out))

    print(causal_mask.flatten()[:10])

