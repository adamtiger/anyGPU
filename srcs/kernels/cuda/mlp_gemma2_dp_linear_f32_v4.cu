#include "mlp_gemma2_dp_linear.cuh"

constexpr int x_width = 9216;
constexpr int w_width = 2304;

constexpr int NUM_WARPS = 8;
constexpr int WARP_SIZE = 32;

constexpr int TS_H = 32;
constexpr int TS_K = 64;
constexpr int TS_W = 32;

constexpr int NR = TS_H / NUM_WARPS;        // number of xt rows, used by a warp
constexpr int NUM_KTILES = x_width / TS_K;  // number tiles along the common dimension
constexpr int NR_LOAD = TS_K / NUM_WARPS;   // during glob -> shared load each warp handles more rows


__global__ void cu_mlp_gemma2_dp_linear_f32_v4_kernel(
	const int sl,
	const float32* dx,
	const float32* dwdp,
	float32* dy)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i0 = blockIdx.y * TS_H;
	int imax = min(TS_H, sl - i0);
	int lane_id = threadIdx.y;

	// outputs can be stored in local memory for speed
	float32 y[NR] = {};
	for (int s = 0; s < NR; ++s)
	{
		y[s] = 0.f;
	}

	// cycle over internal tiles
	for (int tk = 0; tk < NUM_KTILES; ++tk)
	{
		// load the data from glob xt to shared mem
		__shared__ float32 shrd_x[TS_H * TS_K];
		for (int r = 0; r < NR_LOAD; ++r)
		{
			int wrow_id = lane_id / 2;
			int shrd_index_x = (wrow_id * NR_LOAD + r) * TS_K + WARP_SIZE * (lane_id % 2) + threadIdx.x;

			if (wrow_id * NR_LOAD + r < imax)
			{
				int glob_index_x = (i0 + wrow_id * NR_LOAD + r) * x_width + tk * TS_K + WARP_SIZE * (lane_id % 2) + threadIdx.x;
				shrd_x[shrd_index_x] = dx[glob_index_x];
			}
			else
			{
				shrd_x[shrd_index_x] = 0.f;
			}
		}

		// load the data from glob w to shared mem
		__shared__ float32 shrd_w[TS_K * TS_W];
		for (int r = 0; r < NR_LOAD; ++r)
		{
			int glob_index_w = (tk * TS_K + lane_id * NR_LOAD + r) * w_width + j;
			int shrd_index_w = (lane_id * NR_LOAD + r) * TS_W + threadIdx.x;
			shrd_w[shrd_index_w] = dwdp[glob_index_w];
		}

		__syncthreads();

		// calculate matmul
		for (int r = 0; r < NR; r+=4)
		{
			for (int k = 0; k < TS_K; k+=8)
			{
				float32 reg_x[4][8];
				float32 reg_w[8];

				// load from shared to registers
				for (int m = 0; m < 4; ++m)
				{
					for (int n = 0; n < 8; ++n)
					{
						reg_x[m][n] = shrd_x[(lane_id * NR + r + m) * TS_K + k + n];
					}
				}

				for (int n = 0; n < 8; ++n)
				{
					reg_w[n] = shrd_w[(k + n) * TS_W + threadIdx.x];
				}

				// multiply
				for (int m = 0; m < 4; ++m)
				{
					for (int n = 0; n < 8; ++n)
					{
						y[r + m] += reg_x[m][n] * reg_w[n];
					}
				}
			}
		}

		__syncthreads();
	}

	// store results back in the global memory
	for (int r = 0; r < NR; ++r)
	{
		int i = i0 + lane_id * NR + r;
		if (i < sl)
		{
			int glob_index_y = i * w_width + j;
			dy[glob_index_y] = y[r];
		}
	}
}


void cu_mlp_gemma2_dp_linear_f32_v4(
	const Tensor<float32, CUDA>& xt,     // input (batch * seq_len, 9216)
	const Tensor<float32, CUDA>& wt_dp,  // down proj weight, (9216, 2304)
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int sl = xt.shape[0] * xt.shape[1];

	auto* dx = xt.buffer();
	auto* dwdp = wt_dp.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { WARP_SIZE, NUM_WARPS, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, TS_W), calc_req_num_blocks(sl, TS_H), 1 };

	cu_mlp_gemma2_dp_linear_f32_v4_kernel<<<gs, bs>>>(sl, dx, dwdp, dy);
	CUDA_CHECK_LAST_ERROR();
}
