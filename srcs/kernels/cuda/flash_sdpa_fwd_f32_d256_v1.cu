#include "flash_sdpa_fwd.cuh"

constexpr int32 d = 256;

constexpr int32 WARP_SIZE = 32;
constexpr int32 NUM_WARPS = 2;

constexpr int32 TS = 11;  // TS * d * 4(byte) = 45056(byte)
constexpr int32 PS = (TS * TS) / (WARP_SIZE * NUM_WARPS) + 1;  // softmax tile
constexpr int32 PY = d / (WARP_SIZE * NUM_WARPS);  // output tile, num elements to process per thread


__global__ void kernel_cu_flash_sdpa_fwd_d256_v1(
	const float32 alpha,
	const int32 N,
	const float32* dq,
	const float32* dk,
	const float32* dv,
	float32* dy)
{
	// loading corresponding tile from q and y (for current block)
	// global to shared
	__shared__ float32 smem_q[TS][d];
	__shared__ float32 smem_y[TS][d];

	int br = blockIdx.x;
	int tx = threadIdx.x;

	for (int r = 0; r < TS; ++r)  // each thread reads 4 elements (vectorized load)
	{
		*(reinterpret_cast<float4*>(smem_q[r]) + tx) = *(reinterpret_cast<const float4*>(dq + br * TS * d + r * d) + tx);
		*(reinterpret_cast<float4*>(smem_y[r]) + tx) = float4(0);
	}

	// shared mem for maximum and sum values
	__shared__ float32 m_prev[TS];
	__shared__ float32 m_curr[TS];

	__shared__ float32 l_prev[TS];
	__shared__ float32 l_curr[TS];

	for (int i = 0; i < TS; ++i)
	{
		m_prev[i] = -3.4e+38f;
		l_prev[i] = 0;
	}

	// iterate over the k, v matrices (tile-by-tile)
	for (int j = 0; j < gridDim.x; ++j)
	{
		// loading corresponding tile from k and v (for current block)
	    // global to shared
		__shared__ float32 smem_k[TS][d];
		__shared__ float32 smem_v[TS][d];

		for (int r = 0; r < TS; ++r)  // each thread reads 4 elements (vectorized load)
		{
			*(reinterpret_cast<float4*>(smem_k[r]) + tx) = *(reinterpret_cast<const float4*>(dk + j * TS * d + r * d) + tx);
			*(reinterpret_cast<float4*>(smem_v[r]) + tx) = *(reinterpret_cast<const float4*>(dv + j * TS * d + r * d) + tx);
		}
		__syncthreads();

		// calculate Q@K^T * alpha
		__shared__ float32 smem_qk[TS][TS];

		for (int t = tx * PS; t < TS * TS && t < tx * PS + PS; ++t)  // each thread processes P elements
		{
			int r = t / TS;
			int c = t % TS;

			float32 acc = 0;
			for (int k = 0; k < d; ++k)
			{
				acc += smem_q[r][k] * smem_k[c][k];
			}
			smem_qk[r][c] = acc * alpha;  // applying scaling too
		}
		__syncthreads();

		// calculate maximum of the row
		if (tx < TS)
		{
			float32 m = smem_qk[tx][0];
			for (int c = 1; c < TS; ++c)
			{
				m = fmaxf(m, smem_qk[tx][c]);
			}
			m_curr[tx] = fmaxf(m, m_prev[tx]);  // save a multiplication later
		}
		__syncthreads();

		// calculate exponents of the Q@K^T tile
		for (int t = tx * PS; t < TS * TS && t < tx * PS + PS; ++t)  // each thread processes P elements
		{
			int r = t / TS;
			int c = t % TS;

			smem_qk[r][c] = expf(smem_qk[r][c] - m_curr[r]);
		}
		__syncthreads();

		// calculate sum of the row
		if (tx < TS)
		{
			float32 l = smem_qk[tx][0];
			for (int c = 1; c < TS; ++c)
			{
				l += smem_qk[tx][c];
			}
			l_curr[tx] = l + l_prev[tx] * expf(m_prev[tx] - m_curr[tx]);  // save a multiplication later
		}
		__syncthreads();

		// devide exp(Q@K^T)
		for (int t = tx * PS; t < TS * TS && t < tx * PS + PS; ++t)  // each thread processes P elements
		{
			int r = t / TS;
			int c = t % TS;

			smem_qk[r][c] /= l_curr[r];
		}
		__syncthreads();

		// calculate softmax_partial(Q@K^T)@V
		for (int r = 0; r < TS; ++r)
		{
			for (int c = tx * PY; c < d && c < tx * PY + PY; ++c)
			{
				float32 acc = 0;
				for (int k = 0; k < TS; ++k)
				{
					acc += smem_qk[r][k] * smem_v[k][c];
				}

				// compensate according to the m and l values
				// to calculate the final output value
				float32 y_prev = smem_y[r][c];

				float32 y_curr = y_prev * expf(m_prev[r] - m_curr[r]) * l_prev[r] / l_curr[r] + acc;

				smem_y[r][c] = y_curr;
			}
		}
		__syncthreads();

		// set the values of m and l
		if (tx < TS)
		{
			m_prev[tx] = m_curr[tx];
			l_prev[tx] = l_curr[tx];
		}
	}

	// loading corresponding tile from y (for current block)
	// shared to global
	for (int r = 0; r < TS; ++r)  // each thread writes 4 elements (vectorized load)
	{
		*(reinterpret_cast<float4*>(dy + br * TS * d + r * d) + tx) = *(reinterpret_cast<float4*>(smem_y[r]) + tx);
	}
}


void cu_flash_sdpa_fwd_d256_v1(
	const float32 alpha,
	const Tensor<float32, CUDA>& qt,
	const Tensor<float32, CUDA>& kt,
	const Tensor<float32, CUDA>& vt,
	Tensor<float32, CUDA>& yt)
{
	const int32 N = qt.shape[0];
	const float32* dq = qt.buffer();
	const float32* dk = kt.buffer();
	const float32* dv = vt.buffer();
	float32* dy = yt.buffer();

	// kernel launch
	dim3 bs = { WARP_SIZE * NUM_WARPS, 1, 1 };
	dim3 gs = { calc_req_num_blocks(N, TS), 1, 1 };

	kernel_cu_flash_sdpa_fwd_d256_v1<<<gs, bs>>>(alpha, N, dq, dk, dv, dy);
	CUDA_CHECK_LAST_ERROR();
}
