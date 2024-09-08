#include "performance.hpp"
#include "ops.hpp"


void perf_mm_f32_64x128_128x32()
{
	auto dta = crt_random_tensor<float32, CUDA>({ 64, 128 }, 11);
	auto dtb = crt_random_tensor<float32, CUDA>({ 128, 32 }, 48);
	tensor_mm(dta, dtb);
}

void perf_mm_f32_640x1280_1280x320()
{
	auto dta = crt_random_tensor<float32, CUDA>({ 640, 1280 }, 21);
	auto dtb = crt_random_tensor<float32, CUDA>({ 1280, 320 }, 75);
	tensor_mm(dta, dtb);
}
