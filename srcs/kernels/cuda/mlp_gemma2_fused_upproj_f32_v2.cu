#include "mlp_gemma2_fused_upproj.cuh"

constexpr int x_width = 2304;
constexpr int w_width = 9216;

constexpr int NUM_WARPS = 2;
constexpr int WARP_SIZE = 32;

constexpr int TS_W = WARP_SIZE * NUM_WARPS;  // width in weights
constexpr int TS_K = 16;  // common axis (mm)
constexpr int TS_H = 16;  // height in xt
constexpr int NT = 4;     // NT x TS_K is read from xt to provide balance among warps


static __device__ float gelu_approx(const float z)
{
	return z * 0.5f * (1.f + tanhf(sqrt(2.f / 3.14159265358979323846f) * (z + 0.044715f * z * z * z)));
}


__global__ void cu_mlp_gemma2_fused_uprpoj_f32_v2_kernel(
	const int sl,
	const float32* dx,
	const float32* dwgp,
	const float32* dwup,
	float32* dy)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i0 = blockIdx.y * TS_H;
	int imax = min(TS_H, sl - i0);

	// output storage (temporary for intermediate results)
	__shared__ float32 shrd_ymm1[TS_H * TS_W];
	__shared__ float32 shrd_ymm2[TS_H * TS_W];
	for (int i = 0; i < imax; ++i)  // init to zero
	{
		shrd_ymm1[i * TS_W + threadIdx.x] = 0.f;
		shrd_ymm2[i * TS_W + threadIdx.x] = 0.f;
	}

	// iterate over tiles
	int num_common_tiles = x_width / (TS_K * NT);
	for (int tk = 0; tk < num_common_tiles; ++tk)
	{
		// offset due to tiling
		int glob_tile_offset_x = tk * TS_K * NT;
		int glob_tile_offset_w = tk * TS_K * NT * w_width;

		// load glob xt to shared xt tile
		__shared__ float32 shrd_x[TS_H * TS_K * NT];
		for (int i = 0; i < imax; ++i)
		{
			int glob_index_x = glob_tile_offset_x + (i0 + i) * x_width + threadIdx.x;
			int shrd_index_x = i * TS_K * NT + threadIdx.x;
			shrd_x[shrd_index_x] = dx[glob_index_x];
		}

		// iterate over subparts in xt tile
		for (int p = 0; p < NT; ++p)
		{
			// subparts offset
			int glob_subp_offset_w = p * TS_K * w_width;

			// load glob wt_gp
			__shared__ float32 shrd_wgp[TS_K * TS_W];
			for (int k = 0; k < TS_K; ++k)
			{
				int glob_index_w = glob_tile_offset_w + glob_subp_offset_w + k * w_width + j;
				int shrd_index_w = k * TS_W + threadIdx.x;
				shrd_wgp[shrd_index_w] = dwgp[glob_index_w];
			}
			__syncthreads();

			// matmul for tiles (xt @ wt_gp)
			for (int i = 0; i < imax; ++i)
			{
				float32 accumulator = 0.f;
				for (int k = 0; k < TS_K; ++k)
				{
					accumulator += shrd_x[i * TS_K * NT + p * TS_K + k] * shrd_wgp[k * TS_W + threadIdx.x];
				}
				shrd_ymm1[i * TS_W + threadIdx.x] += accumulator;
			}

			// load glob wt_up
			__shared__ float32 shrd_wup[TS_K * TS_W];
			for (int k = 0; k < TS_K; ++k)
			{
				int glob_index_w = glob_tile_offset_w + glob_subp_offset_w + k * w_width + j;
				int shrd_index_w = k * TS_W + threadIdx.x;
				shrd_wup[shrd_index_w] = dwup[glob_index_w];
			}
			__syncthreads();

			// matmul for tiles (xt @ wt_up)
			for (int i = 0; i < imax; ++i)
			{
				float32 accumulator = 0.f;
				for (int k = 0; k < TS_K; ++k)
				{
					accumulator += shrd_x[i * TS_K * NT + p * TS_K + k] * shrd_wup[k * TS_W + threadIdx.x];
				}
				shrd_ymm2[i * TS_W + threadIdx.x] += accumulator;
			}
		}
	}
	__syncthreads();

	// calculate gelu(ymm1) * ymm2
	// store result in global output memory
	for (int i = 0; i < imax; ++i)
	{
		float32 y = gelu_approx(shrd_ymm1[i * TS_W + threadIdx.x]) * shrd_ymm2[i * TS_W + threadIdx.x];
		dy[(i0 + i) * w_width + j] = y;
	}
}


void cu_mlp_gemma2_fused_upproj_f32_v2(
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
	dim3 bs = { TS_W, 1, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, TS_W), calc_req_num_blocks(sl, TS_H), 1 };

	cu_mlp_gemma2_fused_uprpoj_f32_v2_kernel<<<gs, bs >>>(sl, dx, dwgp, dwup, dy);
	CUDA_CHECK_LAST_ERROR();
}
