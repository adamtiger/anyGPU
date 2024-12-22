#include "mlp_gemma2_dp_linear.cuh"

constexpr int BS = 16;    // block size BS X BS
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = (BS * BS) / WARP_SIZE;

constexpr int TS_H = 64;  // output tile, shared mem (but also defines the input tiles)
constexpr int TS_W = 64;  
constexpr int TS_K = 64;

constexpr int RS_H = TS_H / BS;  // register tile
constexpr int RS_W = TS_W / BS;
constexpr int RS_K = TS_K / BS;


__global__ void cu_fast_mm_f32_v1_kernel(
	const int x_width,
	const int w_width,
	const float32* dx,
	const float32* dw,
	float32* dy)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	float32 acc = 0.f;
	for (int k = 0; k < x_width; ++k)
	{
		 acc += dx[i * x_width + k] * dw[k * w_width + j];
	}
	dy[i * w_width + j] = acc;
}


void cu_fast_mm_f32_v1(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA>& wt,
	Tensor<float32, CUDA>& yt)
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

	cu_fast_mm_f32_v1_kernel<<<gs, bs>>>(x_width, w_width, dx, dw, dy);
	CUDA_CHECK_LAST_ERROR();
}
