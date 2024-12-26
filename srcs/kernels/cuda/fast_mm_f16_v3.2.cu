#include "fast_mm.cuh"

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 8;

constexpr int TS = 64;  // tile size
constexpr int NR = TS / NUM_WARPS;
constexpr int NM = TS / 16;


__device__ void calc_perm_indices(
	const int si,  // source
	const int sj,
	int& ti,       // target
	int& tj) 
{
	int li = si;
	int lj = sj / 16;
	
	int li_1 = li % 16;
	int li_0 = li - li_1;
	int lj_1 = lj;

	int li_prime = lj_1 * 4 + li_1 / 4 + li_0;
	int lj_prime = li_1 % 4;

	ti = li_prime;
	tj = lj_prime * 16 + sj % 16;
}


__global__ void cu_fast_mm_f16_v3_2_kernel(
	const int x_width,
	const int w_width,
	const float16* dx,
	const float16* dw,
	float16* dy)
{
	// output tile
	__shared__ float16 shrd_y[TS * TS];
	for (int ix = 0; ix < TS * TS; ++ix)
	{
		shrd_y[ix] = 0.f;
	}

	// iterate over the shared tiles
	int NT = x_width / TS;
	for (int tk = 0; tk < NT; ++tk)
	{
		__shared__ float16 shrd_x[TS * TS];
		__shared__ float16 shrd_w[TS * TS];

		// load global to shared (xt)
		for (int r = 0; r < NR; ++r)
		{
			int glob_idx = (blockIdx.y * TS + threadIdx.y * NR + r) * x_width + tk * TS + threadIdx.x * 2;

			int ti, tj;
			calc_perm_indices(threadIdx.y * NR + r, threadIdx.x * 2, ti, tj);
			int shrd_idx = ti * TS + tj;

			const int32* wide_glob_x = reinterpret_cast<const int32*>(dx);
			int32* wide_shrd_x = reinterpret_cast<int32*>(shrd_x);

			wide_shrd_x[shrd_idx / 2] = wide_glob_x[glob_idx / 2];
		}

		// load global to shared (wt)
		for (int r = 0; r < NR; ++r)
		{
			int glob_idx = (tk * TS + threadIdx.y * NR + r) * w_width + blockIdx.x * TS + threadIdx.x * 2;

			int ti, tj;
			calc_perm_indices(threadIdx.y * NR + r, threadIdx.x * 2, ti, tj);
			int shrd_idx = ti * TS + tj;

			const int32* wide_glob_w = reinterpret_cast<const int32*>(dw);
			int32* wide_shrd_w = reinterpret_cast<int32*>(shrd_w);

			wide_shrd_w[shrd_idx / 2] = wide_glob_w[glob_idx / 2];
		}

		__syncthreads();

		// wgemm based matmul
		constexpr int width = NM * NM / NUM_WARPS;
		int start = (threadIdx.y % 2) * width;
		int row = threadIdx.y / 2;

        #pragma unroll
		for (int c = 0; c < width; ++c)
		{
			int col = c + start;

			nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float16> acc;
			nvcuda::wmma::load_matrix_sync(acc, shrd_y + row * TS * 16 + col * TS * 4, 16, nvcuda::wmma::mem_row_major);

            #pragma unroll
			for (int k = 0; k < NM; ++k)
			{
				nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, float16, nvcuda::wmma::row_major> reg_x;
				nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, float16, nvcuda::wmma::row_major> reg_w;

				nvcuda::wmma::load_matrix_sync(reg_x, shrd_x + row * TS * 16 + k * TS * 4, 16);
				nvcuda::wmma::load_matrix_sync(reg_w, shrd_w + k * TS * 16 + col * TS * 4, 16);

				nvcuda::wmma::mma_sync(acc, reg_x, reg_w, acc);
			}

			nvcuda::wmma::store_matrix_sync(shrd_y + row * TS * 16 + col * TS * 4, acc, 16, nvcuda::wmma::mem_row_major);
		}

		__syncthreads();
	}

	// store from shared to global
	for (int r = 0; r < NR; ++r)
	{
		int glob_idx = (blockIdx.y * TS + threadIdx.y * NR + r) * w_width + blockIdx.x * TS + threadIdx.x * 2;
		int ti, tj;
		calc_perm_indices(threadIdx.y * NR + r, threadIdx.x * 2, ti, tj);
		int shrd_idx = ti * TS + tj;

		int32* wide_glob_y = reinterpret_cast<int32*>(dy);
		int32* wide_shrd_y = reinterpret_cast<int32*>(shrd_y);

		wide_glob_y[glob_idx / 2] = wide_shrd_y[shrd_idx / 2];
	}
}


void cu_fast_mm_f16_v3_2(
	const Tensor<float16, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float16, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float16, CUDA>& yt)
{
	const int x_height = xt.shape[0];
	const int x_width = xt.shape[1];
	const int w_width = wt.shape[1];

	const float16* dx = xt.buffer();
	const float16* dw = wt.buffer();
	float16* dy = yt.buffer();

	dim3 bs = { WARP_SIZE, NUM_WARPS, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, TS), calc_req_num_blocks(x_height, TS), 1 };

	cu_fast_mm_f16_v3_2_kernel<<<gs, bs>>>(x_width, w_width, dx, dw, dy);
	CUDA_CHECK_LAST_ERROR();
}
