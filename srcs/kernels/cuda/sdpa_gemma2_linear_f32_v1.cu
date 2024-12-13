#include "sdpa_gemma2_linear.cuh"


constexpr int NUM_WARPS = 2;
constexpr int WARP_SIZE = 32;

constexpr int TS_H = 16;
constexpr int TS_K = 64;
constexpr int TS_W = 64;


__global__ void cu_sdpa_gemma2_linear_f32_v1_kernel(
	const int sl,
	const int x_width,
	const int w_width,
	const float32* dx,
	const float32* dwdp,
	float32* dy)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i0 = blockIdx.y * TS_H;
	int imax = min(TS_H, sl - i0);
	//int lane_id = threadIdx.x / WARP_SIZE;

	// outputs can be stored in local memory for speed
	__shared__ float32 shrd_y[TS_H * TS_K];
	for (int s = 0; s < TS_H * TS_K; ++s)
	{
		shrd_y[s] = 0.f;
	}

	// cycle over internal tiles
	int num_ktiles = x_width / TS_K;
	for (int tk = 0; tk < num_ktiles; ++tk)
	{
		// load the data from glob xt to shared mem
		__shared__ float32 shrd_x[TS_H * TS_K];
		for (int r = 0; r < TS_H; ++r)
		{
			int shrd_index_x = r * TS_K + threadIdx.x;

			if (r < imax)
			{
				int glob_index_x = (i0 + r) * x_width + tk * TS_K + threadIdx.x;
				shrd_x[shrd_index_x] = dx[glob_index_x];
			}
			else
			{
				shrd_x[shrd_index_x] = 0.f;
			}
		}

		// load the data from glob w to shared mem
		__shared__ float32 shrd_w[TS_K * TS_W];
		for (int r = 0; r < TS_K; ++r)
		{
			int glob_index_w = (tk * TS_K + r) * w_width + j;
			int shrd_index_w = r * TS_W + threadIdx.x;
			shrd_w[shrd_index_w] = dwdp[glob_index_w];
		}

		__syncthreads();

		// calculate matmul
		for (int r = 0; r < TS_H; ++r)
		{
			for (int k = 0; k < TS_K; ++k)
			{
				int index_x = r * TS_K + k;
				int index_w = k * TS_W + threadIdx.x;
				shrd_y[r * TS_K + threadIdx.x] += shrd_x[index_x] * shrd_w[index_w];
			}
		}

		__syncthreads();
	}

	// store results back in the global memory
	for (int r = 0; r < TS_H; ++r)
	{
		int i = i0 + r;
		if (i < sl)
		{
			int glob_index_y = i * w_width + j;
			dy[glob_index_y] = shrd_y[r * TS_W + threadIdx.x];
		}
	}
}


void cu_sdpa_gemma2_linear_f32_v1(
	const Tensor<float32, CUDA>& xt,  // input (batch * seq_len, multiple of 64)
	const Tensor<float32, CUDA>& wt,  // proj weight, multiple of 64
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int sl = xt.shape[0] * xt.shape[1];
	const int x_width = xt.shape[2];
	const int w_width = wt.shape[1];

	auto* dx = xt.buffer();
	auto* dw = wt.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { WARP_SIZE * NUM_WARPS, 1, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, TS_W), calc_req_num_blocks(sl, TS_H), 1 };

	cu_sdpa_gemma2_linear_f32_v1_kernel<<<gs, bs>>>(sl, x_width, w_width, dx, dw, dy);
	CUDA_CHECK_LAST_ERROR();
}
