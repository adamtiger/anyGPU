#ifndef __EXT_TORCH_TESTS__
#define __EXT_TORCH_TESTS__

void external_test_sdp_fwd_f32();
void external_test_sdp_bwd_f32();
void external_test_sdp_masked_scaled_fwd_f32();

void external_test_cpu_softmax_bwd_f32();

void external_test_sf_data_reading();

void external_test_group_norm_fwd_f32();
void external_test_layer_norm_fwd_f32();
void external_test_rms_norm_fwd_f32();

void external_test_silu_fwd_f32();
void external_test_gelu_fwd_f32();
void external_test_gelu_approx_fwd_f32();

void external_test_embedding_fwd_f32();
void external_test_rotary_embedding_fwd_f32();
void external_test_alt_rotary_embedding_fwd_f32();

void external_test_linear_fwd_f32();
void external_test_transpose_fwd_f32();

void external_test_mm_m1024_n2048_k2304_f32();
void external_test_mm_m1024_n2048_k2304_f16();

void external_test_concat_fwd_f32();
void external_test_repeat_fwd_f32();
void external_test_slice_fwd_f32();

void external_test_causal_conv1d_fwd_f32();

void external_test_conv2d_k3x3_s1x1_p0x0_1_fwd_f32();

#endif  // __EXT_TORCH_TESTS__