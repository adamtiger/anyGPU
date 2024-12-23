#include "fast_mm.cuh"


__global__ void cu_fast_mm_f16_v1_kernel(
	const int x_width,
	const int w_width,
	const float16* dx,
	const float16* dw,
	float16* dy)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	float16 acc = 0.f;
	for (int k = 0; k < x_width; ++k)
	{
		acc += dx[i * x_width + k] * dw[k * w_width + j];
	}
	dy[i * w_width + j] = acc;
}


void cu_fast_mm_f16_v1(
	const Tensor<float16, CUDA>& xt,
	const Tensor<float16, CUDA>& wt,
	Tensor<float16, CUDA>& yt)
{
	// calc kernel arguments
	const int x_height = xt.shape[0];
	const int x_width = xt.shape[1];
	const int w_width = wt.shape[1];

	auto* dx = xt.buffer();
	auto* dw = wt.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { 32, 8, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, bs.x), calc_req_num_blocks(x_height, bs.y), 1 };

	cu_fast_mm_f16_v1_kernel<<<gs, bs>>>(x_width, w_width, dx, dw, dy);
	CUDA_CHECK_LAST_ERROR();
}
