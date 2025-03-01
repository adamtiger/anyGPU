#include "fast_mm.cuh"

const int WARP_SIZE = 32;

// register level tile size
//   per-thread tile
const int RH = 8;
const int RW = 4;
const int RK = 2;

// warp level tile size (in output)
const int WH = RH;
const int WW = WARP_SIZE * RW;

// shared mem level tile size
//   per-block tile
const int NUM_WARPS = 8;

const int TH = NUM_WARPS * WH;
const int TW = WW;
const int TK = WARP_SIZE * RK / 2;  // double buffering

// derived numbers
const int ROWS_X = TH / NUM_WARPS;  // global -> shared load
const int ROWS_W = TK / NUM_WARPS;


__global__ void cu_fast_mm_f32_v4_1_kernel(
	const int x_width,
	const int w_width,
	const float32* dx,
	const float32* dw,
	float32* dy)
{
	// define and init helper data
	float x_reg[RH][RK];
	float w_reg[RK][RW];
	float y_reg[RH][RW];

	for (int i = 0; i < RH; ++i)
	{
		for (int j = 0; j < RW; ++j)
		{
			y_reg[i][j] = 0.f;
		}
	}

	__shared__ float x_shared[2][TH * TK];
	__shared__ float w_shared[2][TK * TW];

	// infer thread info
	int bh = blockIdx.y;  // block index
	int bw = blockIdx.x;

	int thread_idx = (threadIdx.y * blockDim.x + threadIdx.x);
	int lane_idx = thread_idx / WARP_SIZE;
	int tidx = thread_idx % WARP_SIZE;

	// load the first tiles from glob to shared
	// load data from global to shared
	int glob_offset_x = bh * TH * x_width + 0 * TK + lane_idx * ROWS_X * x_width;
	int glob_offset_w = 0 * TK * w_width + bw * TW + lane_idx * ROWS_W * w_width;

	int shrd_offset_x = lane_idx * ROWS_X * TK;
	int shrd_offset_w = lane_idx * ROWS_W * TW;

	for (int i = 0; i < ROWS_X; ++i)  // coalesced read (load.64)
	{
		float* sx = x_shared[0] + shrd_offset_x + tidx * RK / 2 + i * TK;
		const float* gx = dx + glob_offset_x + tidx * RK / 2 + i * x_width;
		*reinterpret_cast<float*>(sx) = *reinterpret_cast<const float*>(gx);
	}

	for (int i = 0; i < ROWS_W; ++i)
	{
		float* sw = w_shared[0] + shrd_offset_w + tidx * RW + i * TW;
		const float* gw = dw + glob_offset_w + tidx * RW + i * w_width;
		*reinterpret_cast<float4*>(sw) = *reinterpret_cast<const float4*>(gw);
	}

	__syncthreads();

	// iterate over all the common tiles
	int num_tiles = x_width / TK;
	for (int tk = 1; tk < num_tiles; ++tk)
	{
		// choose compute and load shared tiles
		// for double buffering
		int mem_load_ix = tk % 2;
		int comp_ix = (tk + 1) % 2;

		// load data from global to shared
		int glob_offset_x = bh * TH * x_width + tk * TK + lane_idx * ROWS_X * x_width;
		int glob_offset_w = tk * TK * w_width + bw * TW + lane_idx * ROWS_W * w_width;

		int shrd_offset_x = lane_idx * ROWS_X * TK;
		int shrd_offset_w = lane_idx * ROWS_W * TW;

		for (int i = 0; i < ROWS_X; ++i)  // coalesced read (load.64)
		{
			float* sx = x_shared[mem_load_ix] + shrd_offset_x + tidx * RK / 2 + i * TK;
			const float* gx = dx + glob_offset_x + tidx * RK / 2 + i * x_width;
			*reinterpret_cast<float*>(sx) = *reinterpret_cast<const float*>(gx);
		}

		for (int i = 0; i < ROWS_W; ++i)
		{
			float* sw = w_shared[mem_load_ix] + shrd_offset_w + tidx * RW + i * TW;
			const float* gw = dw + glob_offset_w + tidx * RW + i * w_width;
			*reinterpret_cast<float4*>(sw) = *reinterpret_cast<const float4*>(gw);
		}

		// multiply mm
		for (int rk = 0; rk < TK / RK; ++rk)
		{
			// load from shared to registers
			float* sx = x_shared[comp_ix] + lane_idx * RH * TK + rk * RK;
			for (int r = 0; r < RH; ++r)
			{
				*reinterpret_cast<float2*>(x_reg[r]) = *reinterpret_cast<float2*>(sx + r * TK);
			}

			float* sw = w_shared[comp_ix] + rk * RK * TW + tidx * RW;
			for (int r = 0; r < RK; ++r)
			{
				*reinterpret_cast<float4*>(w_reg[r]) = *reinterpret_cast<float4*>(sw + r * TW);
			}

			// mm on register mm
			for (int i = 0; i < RH; ++i)
			{
				for (int j = 0; j < RW; ++j)
				{
					for (int k = 0; k < RK; ++k)
					{
						y_reg[i][j] += x_reg[i][k] * w_reg[k][j];
					}
				}
			}
		}

		__syncthreads();
	}

	// multiply mm (calculating the last tile)
	for (int rk = 0; rk < TK / RK; ++rk)
	{
		// load from shared to registers
		float* sx = x_shared[1] + lane_idx * RH * TK + rk * RK;
		for (int r = 0; r < RH; ++r)
		{
			*reinterpret_cast<float2*>(x_reg[r]) = *reinterpret_cast<float2*>(sx + r * TK);
		}

		float* sw = w_shared[1] + rk * RK * TW + tidx * RW;
		for (int r = 0; r < RK; ++r)
		{
			*reinterpret_cast<float4*>(w_reg[r]) = *reinterpret_cast<float4*>(sw + r * TW);
		}

		// mm on register mm
		for (int i = 0; i < RH; ++i)
		{
			for (int j = 0; j < RW; ++j)
			{
				for (int k = 0; k < RK; ++k)
				{
					y_reg[i][j] += x_reg[i][k] * w_reg[k][j];
				}
			}
		}
	}

	__syncthreads();

	// load back to global
	int glob_offset_y = bh * TH * w_width + bw * TW + lane_idx * RH * w_width + tidx * RW;
	for (int r = 0; r < RH; ++r)
	{
		float* gy = dy + glob_offset_y + r * w_width;
		*reinterpret_cast<float4*>(gy) = *reinterpret_cast<float4*>(y_reg[r]);
	}
}


void cu_fast_mm_f32_v4_1(
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
	dim3 bs = { NUM_WARPS, WARP_SIZE, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, TW), calc_req_num_blocks(x_height, TH), 1 };

	cu_fast_mm_f32_v4_1_kernel<<<gs, bs>>>(x_width, w_width, dx, dw, dy);
	CUDA_CHECK_LAST_ERROR();
}
