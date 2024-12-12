#include "mlp_gemma2_fused_upproj.cuh"

constexpr int x_width = 2304;
constexpr int w_width = 9216;

constexpr int NUM_WARPS = 4;
constexpr int WARP_SIZE = 32;

constexpr int TS_X = NUM_WARPS * WARP_SIZE;  // number of cols from wt_gp and wt_up, handle at once
constexpr int TS_Y = 16;  // number of rows from xt, to handle at once


__device__ float gelu_approx(const float z)
{
	return z * 0.5f * (1.f + tanhf(sqrt(2.f / 3.14159265358979323846f) * (z + 0.044715f * z * z * z)));
}


__global__ void cu_mlp_gemma2_fused_uprpoj_f32_v1_kernel(
	const int sl,
	const float32* dx,
	const float32* dwgp,
	const float32* dwup,
	float32* dy)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i0 = blockIdx.y * TS_Y;
	int imax = min(TS_Y, sl - i0);

	// cycle over the row of xt
	for (int i = 0; i < imax; ++i)
	{
		float32 ymm1 = 0.f;
		float32 ymm2 = 0.f;

		for (int k = 0; k < x_width; ++k)
		{
			float32 x = dx[(i0 + i) * x_width + k];

			float32 w1 = dwgp[k * w_width + j];
			ymm1 += x * w1;

			float32 w2 = dwup[k * w_width + j];
			ymm2 += x * w2;
		}

		ymm1 = gelu_approx(ymm1);

		float32 y = ymm1 * ymm2;
		dy[(i0 + i) * w_width + j] = y;
	}
}


void cu_mlp_gemma2_fused_upproj_f32_v1(
	const Tensor<float32, CUDA>& xt,     // input (batch * seq_len, 2304)
	const Tensor<float32, CUDA>& wt_gp,  // gate proj weight, (2304, 9216)
	const Tensor<float32, CUDA>& wt_up,  // gate up proj weight, (2304, 9216) 
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int sl = xt.shape[0] * xt.shape[1];

	auto* dx = xt.buffer();
	auto* dwgp = wt_gp.buffer();
	auto* dwup = wt_up.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { TS_X, 1, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, TS_X), calc_req_num_blocks(sl, TS_Y), 1 };

	cu_mlp_gemma2_fused_uprpoj_f32_v1_kernel<<<gs, bs>>>(sl, dx, dwgp, dwup, dy);
	CUDA_CHECK_LAST_ERROR();
}
