#ifndef __FAST_MM_CUH__
#define __FAST_MM_CUH__

#include "tensor.hpp"

// baseline global
void cu_fast_mm_f32_v1(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt);

// tiled matrix multiplication, shared mem level
// register tiled mm
void cu_fast_mm_f32_v3(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt);

#endif  // __FAST_MM_CUH__
