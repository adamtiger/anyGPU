#ifndef __EXT_ZAMBA2_TESTS__
#define __EXT_ZAMBA2_TESTS__

void external_test_zamba2_model_rmsnorm();
void external_test_zamba2_attn_rotary();
void external_test_zamba2_attn_sdp();
void external_test_zamba2_attndeco_mlp();
void external_test_zamba2_attndeco_attn();

void external_test_zamba2_model_attndecoder();

void external_test_mamba2_layer_causal_conv1d();
void external_test_mamba2_layer_gated_rmsnorm();

void test_zamba2_glu();

#endif  // __EXT_ZAMBA2_TESTS__
