#include "fast_mm.cuh"

//#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>


constexpr int NUM_WARPS = 16;
constexpr int WARP_SIZE = 32;

constexpr int TS_H = 64;
constexpr int TS_K = 64;
constexpr int TS_W = 64;

constexpr int NR = TS_H / NUM_WARPS * (TS_W / WARP_SIZE);        // number of xt rows, used by a warp
constexpr int NR2 = NR / 2;

__global__ void cu_fast_mm_f32_v2_1_kernel(
	const int x_width,
	const int w_width,
	const float32* dx,
	const float32* dw,
	float32* dy)
{

	int NUM_KTILES = x_width / TS_K;  // number tiles along the common dimension

	int lane_id = threadIdx.y;
	int j = blockIdx.x * TS_W + threadIdx.x + WARP_SIZE * (lane_id % 2);
	int i0 = blockIdx.y * TS_H;

	// outputs can be stored in local memory for speed
	float32 y[NR] = {};
	for (int s = 0; s < NR; ++s)
	{
		y[s] = 0.f;
	}

	// cycle over internal tiles
	auto block = cooperative_groups::this_thread_block();

	__shared__ float32 shrd_x[TS_H * TS_K];
	__shared__ float32 shrd_w[TS_K * TS_W];
    #pragma unroll 2
	for (int tk = 0; tk < NUM_KTILES; ++tk)
	{

		// load the data from glob xt to shared mem
		// buffer 0 (horizontally contains the left half)
		for (int r = 0; r < NR2; ++r)
		{
			int wrow_id = lane_id;
			int shrd_index_x = (wrow_id * NR2 + r) * TS_K + threadIdx.x;

			int glob_index_x = (i0 + wrow_id * NR2 + r) * x_width + tk * TS_K + threadIdx.x;
			//shrd_x[shrd_index_x] = dx[glob_index_x];
			cooperative_groups::memcpy_async(cooperative_groups::this_thread(), shrd_x + shrd_index_x, dx + glob_index_x, sizeof(float32));
		}

		// load the data from glob w to shared mem
		// buffer 0 (vertically contains the upper half)
		for (int r = 0; r < NR2; ++r)
		{
			int wrow_id = lane_id / 2;
			int shrd_index_w = (wrow_id * NR2 + r) * TS_W + WARP_SIZE * (lane_id % 2) + threadIdx.x;

			int glob_index_w = (tk * TS_K + wrow_id * NR2 + r) * w_width + j;
			//shrd_w[shrd_index_w] = dw[glob_index_w];
			cooperative_groups::memcpy_async(cooperative_groups::this_thread(), shrd_w + shrd_index_w, dw + glob_index_w, sizeof(float32));
		}

		//__syncthreads();
		cooperative_groups::wait(block);

		// load the data from glob xt to shared mem
		// buffer 1 (horizontally contains the right half)
		for (int r = 0; r < NR2; ++r)
		{
			int wrow_id = lane_id;
			int shrd_index_x = (wrow_id * NR2 + r) * TS_K + WARP_SIZE + threadIdx.x;

			int glob_index_x = (i0 + wrow_id * NR2 + r) * x_width + tk * TS_K + WARP_SIZE + threadIdx.x;
			shrd_x[shrd_index_x] = dx[glob_index_x];
			cooperative_groups::memcpy_async(cooperative_groups::this_thread(), shrd_x + shrd_index_x, dx + glob_index_x, sizeof(float32));
		}

		// load the data from glob w to shared mem
		// buffer 1 (vertically contains the bottom half)
		for (int r = 0; r < NR2; ++r)
		{
			int wrow_id = lane_id / 2 + NUM_WARPS / 2;
			int shrd_index_w = (wrow_id * NR2 + r) * TS_W + WARP_SIZE * (lane_id % 2) + threadIdx.x;

			int glob_index_w = (tk * TS_K + wrow_id * NR2 + r) * w_width + j;
			//shrd_w[shrd_index_w] = dw[glob_index_w];
			cooperative_groups::memcpy_async(cooperative_groups::this_thread(), shrd_w + shrd_index_w, dw + glob_index_w, sizeof(float32));
		}

		// calculate matmul
		for (int r = 0; r < NR; ++r)
		{
			for (int k = 0; k < TS_K / 2; ++k)
			{
				int wrow_id = lane_id / 2;

				int index_x = (wrow_id * NR + r) * TS_K + k;
				int index_w = k * TS_W + WARP_SIZE * (lane_id % 2) + threadIdx.x;
				y[r] += shrd_x[index_x] * shrd_w[index_w];
			}
		}

		//__syncthreads();
		cooperative_groups::wait(block);

		// calculate matmul
		for (int r = 0; r < NR; ++r)
		{
			for (int k = TS_K / 2; k < TS_K; ++k)
			{
				int wrow_id = lane_id / 2;

				int index_x = (wrow_id * NR + r) * TS_K + k;
				int index_w = k * TS_W + WARP_SIZE * (lane_id % 2) + threadIdx.x;
				y[r] += shrd_x[index_x] * shrd_w[index_w];
			}
		}
	}

	__syncthreads();

	// store results back in the global memory
	for (int r = 0; r < NR; ++r)
	{
		int wrow_id = lane_id / 2;

		int i = i0 + wrow_id * NR + r;
		int glob_index_y = i * w_width + j;
		dy[glob_index_y] = y[r];
	}
}


void cu_fast_mm_f32_v2_1(
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
	dim3 bs = { WARP_SIZE, NUM_WARPS, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, TS_W), calc_req_num_blocks(x_height, TS_H), 1 };

	cu_fast_mm_f32_v2_1_kernel<<<gs, bs >>>(x_width, w_width, dx, dw, dy);
	CUDA_CHECK_LAST_ERROR();
}
