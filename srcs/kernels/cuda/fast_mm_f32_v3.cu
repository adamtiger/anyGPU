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


__global__ void cu_fast_mm_f32_v3_kernel(
	const int x_width,
	const int w_width,
	const float32* dx,
	const float32* dw,
	float32* dy)
{
	// output data
	float32 rmem_out[RS_H * RS_W];
	for (int ix = 0; ix < RS_H * RS_W; ++ix)
	{
		rmem_out[ix] = 0.f;
	}

	// load data to shared tiles from global memory
	int tid = blockDim.x * threadIdx.y + threadIdx.x;
	int laneid = tid / WARP_SIZE;
	int tx = tid % WARP_SIZE;

	int num_tiles = x_width / TS_K;
	for (int tk = 0; tk < num_tiles; ++tk)  // over shared mem tiles
	{
		// load from global memory to shared tiles
		__shared__ float32 shrd_tile_x[TS_H * TS_K];
		__shared__ float32 shrd_tile_w[TS_K * TS_W];

		int gmem_tile_offset_x = blockIdx.y * TS_H * x_width + tk * TS_K;
		int gmem_tile_offset_w = tk * TS_K * w_width + blockIdx.x * TS_W;

		int NR = TS_H / NUM_WARPS;
		for (int r = 0; r < NR; ++r)  // over rows
		{
			for (int c = 0; c < TS_K / WARP_SIZE; ++c)  // over chnuks with warp size
			{
				int shrd_idx = (laneid * NR + r) * TS_K + c * WARP_SIZE + tx;
				int glob_idx = gmem_tile_offset_x + (laneid * NR + r) * x_width + c * WARP_SIZE + tx;
				shrd_tile_x[shrd_idx] = dx[glob_idx];
			}
		}

		NR = TS_K / NUM_WARPS;
		for (int r = 0; r < NR; ++r)  // over rows
		{
			for (int c = 0; c < TS_W / WARP_SIZE; ++c)  // over chnuks with warp size
			{
				if (c * WARP_SIZE + tx < TS_W)
				{
					int shrd_idx = (laneid * NR + r) * TS_W + c * WARP_SIZE + tx;
					int glob_idx = gmem_tile_offset_w + (laneid * NR + r) * w_width + c * WARP_SIZE + tx;
					shrd_tile_w[shrd_idx] = dw[glob_idx];
				}
			}
		}
		__syncthreads();

		// matmul on tile
		for (int rk = 0; rk < TS_K / RS_K; ++rk)  // over register tiles
		{
			int smem_rtile_offset_x = threadIdx.y * RS_H * TS_K + rk * RS_K;
			int smem_rtile_offset_w = rk * RS_K * TS_W + threadIdx.x * RS_W;

			float32 rmem_x[RS_H * RS_K];
			float32 rmem_w[RS_K * RS_W];

			for (int m = 0; m < RS_H; ++m)
			{
				for (int k = 0; k < RS_K; ++k)
				{
					rmem_x[m * RS_K + k] = shrd_tile_x[smem_rtile_offset_x + m * TS_K + k];
				}
			}

			for (int k = 0; k < RS_K; ++k)
			{
				for (int n = 0; n < RS_W; ++n)
				{
					rmem_w[k * RS_W + n] = shrd_tile_w[smem_rtile_offset_w + k * TS_W + n];
				}
			}

			for (int m = 0; m < RS_H; ++m)
			{
				for (int n = 0; n < RS_W; ++n)
				{
					float32 acc = 0.f;
					for (int k = 0; k < RS_K; ++k)
					{
						acc += rmem_x[m * RS_K + k] * rmem_w[k * RS_W + n];
					}
					rmem_out[m * RS_W + n] += acc;
				}
			}
		}
		__syncthreads();
	}

	// load data to global memory
	int gmem_rtile_offset_y = (blockIdx.y * TS_H + threadIdx.y * RS_H) * w_width + blockIdx.x * TS_W + threadIdx.x * RS_W;
	for (int m = 0; m < RS_H; ++m)
	{
		for (int n = 0; n < RS_W; ++n)
		{
			dy[gmem_rtile_offset_y + m * w_width + n] = rmem_out[m * RS_W + n];
		}
	}
}


void cu_fast_mm_f32_v3(
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
	dim3 bs = { BS, BS, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, TS_W), calc_req_num_blocks(x_height, TS_H), 1 };

	cu_fast_mm_f32_v3_kernel<<<gs, bs>>>(x_width, w_width, dx, dw, dy);
	CUDA_CHECK_LAST_ERROR();
}
