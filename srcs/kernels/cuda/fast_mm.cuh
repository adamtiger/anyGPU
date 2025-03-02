#ifndef __FAST_MM_CUH__
#define __FAST_MM_CUH__

#include "tensor.hpp"

// baseline cublas
void cu_fast_mm_f32_cb(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt);

// baseline global
void cu_fast_mm_f32_v1(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt);


// shared tiles
void cu_fast_mm_f32_v2(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt);

// shared tiles
// double buffering
void cu_fast_mm_f32_v2_1(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt);


// tiled matrix multiplication, shared mem level
// register tiled mm
void cu_fast_mm_f32_v3(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt);


// generic size, tiled mm, warp tiled mm, registered tile
void cu_fast_mm_f32_v4(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt);


// generic size, tiled mm, warp tiled mm, registered tile
// applies double buffering too
void cu_fast_mm_f32_v4_1(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt);


// generic size, tiled mm, warp tiled mm, registered tile
// applies double buffering too, transposed A
void cu_fast_mm_f32_v4_2(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt);


/* float16 versions */

// baseline global
void cu_fast_mm_f16_v1(
	const Tensor<float16, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float16, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float16, CUDA>& yt);


// shared tiles
void cu_fast_mm_f16_v2(
	const Tensor<float16, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float16, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float16, CUDA>& yt);


// shared tiles and wgemm for register level mm
void cu_fast_mm_f16_v3(
	const Tensor<float16, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float16, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float16, CUDA>& yt);

// shared tiles and wgemm for register level mm
// padded shared mem, to dicrease bank conflict
void cu_fast_mm_f16_v3_1(
	const Tensor<float16, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float16, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float16, CUDA>& yt);

// shared tiles and wgemm for register level mm
// permutated shared mem, to dicrease bank conflict
void cu_fast_mm_f16_v3_2(
	const Tensor<float16, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float16, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float16, CUDA>& yt);

#endif  // __FAST_MM_CUH__
