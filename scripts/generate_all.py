from sdp_test_cases import * 
from sf_test_cases import *
from ops_test_cases import *
from fused_test_cases import *
from mm_test_cases import generate_all_mm
from gemma2 import slide_mask_test
from gemma2 import fused_mlp_test

if __name__ == '__main__':

    path = r"C:\Data\AI\projects\anyGPU\artifacts"

    # generate_sdp_fwd_nomask_noscore(path, "test_sdp_fwd_nomask_noscore_f32_16_64", 16, 64)
    # generate_sdp_bwd_nomask_noscore(path, "test_sdp_bwd_nomask_noscore_f32_16_64", 16, 64)
    # generate_softmax_bwd(path, "test_softmax_bwd_f32_16_64", 16, 64)
    # generate_sdp_fwd_masked_scaled(path, "test_sdp_fwd_masked_scaled_f32_16_128", 16, 128)
    generate_sdp_fwd_nomask_noscore(path, "test_sdp_fwd_nomask_noscore_f32_2640_256", 2640, 256)

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

    #slide_mask_test.generate_slide_mask_f32(path, "test_gemma2_slide_mask_f32")
    # fused_mlp_test.generate_fused_mlp_upproj_f32(path, "test_gemma2_fused_mlp_upproj_f32", 1)
    # fused_mlp_test.generate_fused_mlp_upproj_f32(path, "test_gemma2_fused_mlp_upproj_f32", 9)
    # fused_mlp_test.generate_fused_mlp_upproj_f32(path, "test_gemma2_fused_mlp_upproj_f32", 158)

    #generate_all_mm(path)
