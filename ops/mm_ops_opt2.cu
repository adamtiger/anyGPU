#include "mm_ops.cuh"

using namespace nvcuda;

/* opt2 implementation */

constexpr int TS = 64;  // tile size (square)
constexpr int NR = 4;   // rows handled by a warp
constexpr int NC = 2;   // cols handled by a warp
constexpr int WS = 32;  // warp size

__global__ void tensor_mm_kernel_f16_opt2(
	const int m, const int n, const int k,
	const float16* dlhs, const float16* drhs,
	float16* dout)
{
	// thread index info
	// warp_index = threadIdx.y;
	// lane_index = threadIdx.x;

	// calculate gmem base offset for current thread block (lhs, rhs, out)
	int gmem_lhs_base = blockIdx.y * TS * k;
	int gmem_rhs_base = blockIdx.x * TS;
	int gmem_out_base = blockIdx.y * TS * n + blockIdx.x * TS;

	// initiate registers for storing the outputs for current thread
	// 32 x 16 threads is responsible for 64 x 64 values -> 8 value per thread
	float16 outputs[NR][NC] = { {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.} };

	// iterate over the tiles corresponding to the current thread block

	int n_tiles = k / TS;  // number of tiles to compute for an output
	for (int tile_ix = 0; tile_ix < n_tiles; ++tile_ix)
	{
		// calculate tile base offsets in gmem (lhs, rhs, out is independent)
		int gmem_lhs_tile = gmem_lhs_base + tile_ix * TS;
		int gmem_rhs_tile = gmem_rhs_base + tile_ix * TS * n;

		// load data to smem from gmem
		__shared__ float16 smem_lhs_store[TS][TS];
		__shared__ float16 smem_rhs_store[TS][TS];

		for (int r = 0; r < NR; ++r)
		{
			for (int c = 0; c < NC; ++c)
			{
				int tile_thread_offset = (threadIdx.y * NR + r) * k + c * WS + threadIdx.x;
				smem_lhs_store[threadIdx.y * NR + r][c * WS + threadIdx.x] = dlhs[gmem_lhs_tile + tile_thread_offset];
			}
		}

		for (int r = 0; r < NR; ++r)
		{
			for (int c = 0; c < NC; ++c)
			{
				int tile_thread_offset = (threadIdx.y * NR + r) * n + c * WS + threadIdx.x;
				smem_rhs_store[threadIdx.y * NR + r][c * WS + threadIdx.x] = drhs[gmem_rhs_tile + tile_thread_offset];
			}
		}

		__syncthreads();

		wmma::fragment<wmma::accumulator, 16, 16, 16, float32> frag;

		// calculate matmul for current tile
		for (int r = 0; r < NR; ++r)
		{
			for (int c = 0; c < NC; ++c)
			{
				for (int z = 0; z < TS; ++z)
				{
					float16 lhs_val = smem_lhs_store[threadIdx.y * NR + r][z];
					float16 rhs_val = smem_rhs_store[z][c * WS + threadIdx.x];
					outputs[r][c] = __hadd(__hmul(lhs_val, rhs_val), outputs[r][c]);
				}
			}
		}

		__syncthreads();
	}

	// load result back to glob memory (should be coalesced write)
	for (int r = 0; r < NR; ++r)
	{
		for (int c = 0; c < NC; ++c)
		{
			int tile_thread_offset = (threadIdx.y * NR + r) * n + c * WS + threadIdx.x;
			dout[gmem_out_base + tile_thread_offset] = outputs[r][c];
		}
	}
}


void tensor_mm_f16_opt2(
	const Tensor<float16, CUDA>& lhs,
	const Tensor<float16, CUDA>& rhs,
	const Tensor<float16, CUDA>& out)
{
	int m = lhs.shape[0];
	int n = rhs.shape[1];
	int k = lhs.shape[1];

	float16* dlhs = lhs.buffer();
	float16* drhs = rhs.buffer();
	float16* dout = out.buffer();

	dim3 bs = { 32, 16, 1 };  // rtx3050 can use up to 3 blocks per sm

	unsigned int gsx = calc_req_num_blocks(n, TS);  // horizontal
	unsigned int gsy = calc_req_num_blocks(m, TS);  // vertical
	dim3 gs = { gsx, gsy, 1 };

	tensor_mm_kernel_f16_opt2<<<gs, bs>>>(m, n, k, dlhs, drhs, dout);
	cudaDeviceSynchronize();
	CUDA_CHECK_LAST_ERROR();
}
