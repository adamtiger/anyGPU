#include "mm_ops.cuh"

/* opt0 implementation */

__global__ void tensor_mm_kernel_f32_opt0(const int m, const int n, const int k, const float32* dlhs, const float32* drhs, float32* dout)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tx < m && ty < n)
	{
		int offset_out = tx * n + ty;

		float32 acc = 0.f;

		for (int l = 0; l < k; ++l)
		{
			int offset_lhs = tx * k + l;
			int offset_rhs = l * n + ty;
			acc += dlhs[offset_lhs] * drhs[offset_rhs];
		}

		dout[offset_out] = acc;
	}
}


void tensor_mm_f32_opt0(
	const Tensor<float32, CUDA>& lhs,
	const Tensor<float32, CUDA>& rhs,
	const Tensor<float32, CUDA>& out)
{
	int m = lhs.shape[0];
	int n = rhs.shape[1];
	int k = lhs.shape[1];

	float32* dlhs = lhs.buffer();
	float32* drhs = rhs.buffer();
	float32* dout = out.buffer();

	dim3 bs = {16, 16, 1};

	unsigned int gsx = m / bs.x + ((m % bs.x > 0) ? 1 : 0);  // vertical
	unsigned int gsy = n / bs.y + ((n % bs.y > 0) ? 1 : 0);  // horizontal
	dim3 gs = {gsx, gsy, 1};

	tensor_mm_kernel_f32_opt0<<<gs, bs>>>(m, n, k, dlhs, drhs, dout);
}





/* opt1 implementation */

constexpr int TS = 64;  // tile size (square)
constexpr int NR = 4;   // rows handled by a warp
constexpr int NC = 2;   // cols handled by a warp
constexpr int WS = 32;  // warp size

__global__ void tensor_mm_kernel_f32_opt1(
	const int m, const int n, const int k, 
	const float32* dlhs, const float32* drhs, 
	float32* dout)
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
	float32 outputs[NR][NC] = { {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f} };

	// iterate over the tiles corresponding to the current thread block

	int n_tiles = k / TS;  // number of tiles to compute for an output
	for (int tile_ix = 0; tile_ix < n_tiles; ++tile_ix)
	{
		// calculate tile base offsets in gmem (lhs, rhs, out is independent)
		int gmem_lhs_tile = gmem_lhs_base + tile_ix * TS;
		int gmem_rhs_tile = gmem_rhs_base + tile_ix * TS * n;

		// load data to smem from gmem
		__shared__ float32 smem_lhs_store[TS][TS];
		__shared__ float32 smem_rhs_store[TS][TS];

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

		// calculate matmul for current tile
		for (int r = 0; r < NR; ++r)
		{
			for (int c = 0; c < NC; ++c)
			{
				for (int z = 0; z < TS; ++z)
				{
					float32 lhs_val = smem_lhs_store[threadIdx.y * NR + r][z];
					float32 rhs_val = smem_rhs_store[z][c * WS + threadIdx.x];
					outputs[r][c] += lhs_val * rhs_val;
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


void tensor_mm_f32_opt1(
	const Tensor<float32, CUDA>& lhs,
	const Tensor<float32, CUDA>& rhs,
	const Tensor<float32, CUDA>& out)
{
	int m = lhs.shape[0];
	int n = rhs.shape[1];
	int k = lhs.shape[1];

	float32* dlhs = lhs.buffer();
	float32* drhs = rhs.buffer();
	float32* dout = out.buffer();

	dim3 bs = { 32, 16, 1 };  // rtx3050 can use up to 3 blocks per sm

	unsigned int gsx = calc_req_num_blocks(n, TS);  // vertical
	unsigned int gsy = calc_req_num_blocks(m, TS);  // horizontal
	dim3 gs = { gsx, gsy, 1 };

	tensor_mm_kernel_f32_opt1<<<gs, bs>>>(m, n, k, dlhs, drhs, dout);
	cudaDeviceSynchronize();
	std::cout << cudaGetLastError() << std::endl;
}
