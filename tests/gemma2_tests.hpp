#ifndef __EXT_GEMMA2_TESTS__
#define __EXT_GEMMA2_TESTS__

void external_test_gemma2_decoder_mlp();
void external_test_gemma2_decoder_rmsn();  // input_rmsns
void external_test_gemma2_decoder_attention();
void external_test_gemma2_attention_rotary();
void external_test_gemma2_model_decoder();

void external_test_gemma2_lmhead_softcap();
void external_test_gemma2_update_mask();
void external_test_gemma2_kvcache_update();

void external_test_gemma2_model_decoder_15();
void external_test_gemma2_model_decoder_16();

void external_test_gemma2_slide_mask();

void external_test_gemma2_causallm();

// more tests for internal functions
void external_test_gemma2_decoder_fused_mlp();

#endif  // __EXT_GEMMA2_TESTS__
