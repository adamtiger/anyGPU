#ifndef __TESTS__
#define __TESTS__

#include "ext_torch_tests.hpp"

void test_binary_add_f32();
void test_binary_add_i32();

void test_binary_mul_f32();
void test_binary_mul_i32();


void test_mm_f32();
void test_mm_f32_640x1280_1280x320();
void test_mm_f16_640x1280_1280x320();


void test_transp_f32();


void test_softmax_f32();
void test_softmax_bwd_f32();


void test_sdp_fwd_f32();
void test_sdp_bwd_f32();
void test_quant_sdp_fwd_f32_i8();


void test_quant_lin_f32_i8();
void test_dequant_lin_i8_f32();
void test_qmm_i8_f32();


#endif  // __TESTS__
