#ifndef __EXT_TORCH_TESTS__
#define __EXT_TORCH_TESTS__

void external_test_sdp_fwd_f32();
void external_test_sdp_bwd_f32();

void external_test_cpu_softmax_bwd_f32();

void external_test_sf_data_reading();

void external_test_layer_norm_fwd_f32();

#endif  // __EXT_TORCH_TESTS__