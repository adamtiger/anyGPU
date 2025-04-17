#include "norm_ops.cuh"

static __device__ int calc_nk(const int region_size)
{
	int nk = region_size / warpSize;
	nk += (region_size % warpSize > 0 ? 1 : 0);
	return nk;
}

static __device__ float32 sum_reduce_warp(const float32 thread_value)
{
	float32 reduced = thread_value;
	for (int k = 16; k >= 1; k = k / 2)
	{
		reduced += __shfl_xor_sync(0xFFFFFFFF, reduced, k);
	}
	return reduced;
}

__constant__ float32 c_dw[1344];  // unused in this current settings

__global__ void cu_tensor_rms_norm_f32_v1_kernel(
	const int num_norm_regions,
	const int norm_region_size,
	const float32* dx,
	const float32* dw,
	const float32 eps,
	const float32 delta,
	float32* dy)
{
	// mapping the y block index and thread index to the parallel regions
	int r = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ float32 shr_mem_x[1344];  // fixed size; can be dynamic shr
	__shared__ float32 shr_mem_acc_sum[8];

	if (r < num_norm_regions)
	{
		// offsets, constants
		const int tx = threadIdx.x % warpSize;
		const int warp_id = threadIdx.x / warpSize;
		const int warp_norm_reg_size = norm_region_size / 8;
		const int gmem_base_offs = r * norm_region_size;
		const int gmem_warp_offs = warp_id * warp_norm_reg_size;
		const int nk = calc_nk(warp_norm_reg_size);  // each warp is responsible for a sub-region

		// calculate the mean and variance for a region to be normalized
		float32 accum_sum_sqr = 0.f;

		//   iterate the warp over the region (sum the elements corresponding to this thread)
		for (int k = 0; k < nk; ++k)
		{
			const int x_rel_offs = k * warpSize + tx;
			if (x_rel_offs < warp_norm_reg_size)
			{
				float32 x = dx[gmem_base_offs + gmem_warp_offs + x_rel_offs];
				accum_sum_sqr += x * x;
				shr_mem_x[gmem_warp_offs + x_rel_offs] = x;
			}
		}

		//   reduce accross the warp 
		shr_mem_acc_sum[warp_id] = sum_reduce_warp(accum_sum_sqr);

		__syncthreads();

		// calculate accum sum
		accum_sum_sqr = 0.f;
		for (int i = 0; i < 8; ++i)
		{
			accum_sum_sqr += shr_mem_acc_sum[i];
		}


		//   mean
		const float32 mean_sum_sqr = accum_sum_sqr / static_cast<float32>(norm_region_size);


		// calculate the normalized x values and store them in the output
		for (int k = 0; k < nk; ++k)
		{
			const int x_rel_offs = k * warpSize + tx;
			if (x_rel_offs < warp_norm_reg_size)
			{
				float32 x = shr_mem_x[gmem_warp_offs + x_rel_offs];
				float32 w = dw[gmem_warp_offs + x_rel_offs] + delta;
				float32 y = (x * rsqrtf(mean_sum_sqr + eps)) * w;
				dy[gmem_base_offs + gmem_warp_offs + x_rel_offs] = y;
			}
		}
	}
}

void cu_tensor_rms_norm_f32_v1(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA> wt,
	const float32 eps,
	const bool zero_centered,
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int norm_region_size = wt.size();  // e.g. w shape: [4, 8] -> 32
	const int num_norm_regions = xt.size() / norm_region_size;  // e.g. x shape: [3, 4, 8] -> 3 

	auto* dx = xt.buffer();
	auto* dw = wt.buffer();
	auto* dy = yt.buffer();

	cudaMemcpyToSymbol(c_dw, dw, norm_region_size * sizeof(float32), 0, cudaMemcpyDeviceToDevice);

	float32 delta = (zero_centered ? 1.f : 0.f);

	// kernel lauch params
	dim3 bs = { 32 * 8, 1, 1 };
	dim3 gs = { 1, calc_req_num_blocks(num_norm_regions, bs.y), 1 };

	cu_tensor_rms_norm_f32_v1_kernel<<<gs, bs>>>(num_norm_regions, norm_region_size, dx, dw, eps, delta, dy);
	CUDA_CHECK_LAST_ERROR();
}
