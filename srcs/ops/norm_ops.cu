#include "norm_ops.cuh"

__device__ int calc_nk(const int region_size)
{
	int nk = region_size / warpSize;
	nk += (region_size % warpSize > 0 ? 1 : 0);
	return nk;
}

__device__ float32 sum_reduce_warp(const float32 thread_value)
{
	float32 reduced = thread_value;
	for (int k = 16; k >= 1; k = k / 2)
	{
		reduced += __shfl_xor_sync(0xFFFFFFFF, reduced, k);
	}
	return reduced;
}

__global__ void cu_tensor_layer_norm_f32_kernel(
	const int num_norm_regions,
	const int norm_region_size,
	const float32* dx,
	const float32* dw,
	const float32* db,
	const float32 eps,
	float32* dy)
{
	// mapping the y block index and thread index to the parallel regions
	int r = blockDim.y * blockIdx.y + threadIdx.y;

	if (r < num_norm_regions)
	{
		// offsets, constants
		const int gmem_base_offs = r * norm_region_size;
		const int nk = calc_nk(norm_region_size);

		// calculate the mean and variance for a region to be normalized
		float32 accum_sum = 0.f;
		float32 accum_sum_sqr = 0.f;

		//   iterate the warp over the region (sum the elements corresponding to this thread)
		for (int k = 0; k < nk; ++k)
		{
			const int x_rel_offs = k * warpSize + threadIdx.x;
			if (x_rel_offs < norm_region_size)
			{
				float32 x = dx[gmem_base_offs + x_rel_offs];
				accum_sum += x;
				accum_sum_sqr += x * x;
			}
		}

		//   reduce accross the warp 
		accum_sum = sum_reduce_warp(accum_sum);
		accum_sum_sqr = sum_reduce_warp(accum_sum_sqr);

		//   mean and variance
		const float32 mean = accum_sum / static_cast<float32>(norm_region_size);
		const float32 var = accum_sum_sqr / static_cast<float32>(norm_region_size) - mean * mean;


		// calculate the normalized x values and store them in the output
		for (int k = 0; k < nk; ++k)
		{
			const int x_rel_offs = k * warpSize + threadIdx.x;
			if (x_rel_offs < norm_region_size)
			{
				float32 x = dx[gmem_base_offs + x_rel_offs];
				float32 w = dw[x_rel_offs];
				float32 b = db[x_rel_offs];
				float32 y = ((x - mean) * rsqrtf(var + eps)) * w + b;
				dy[gmem_base_offs + x_rel_offs] = y;
			}
		}
	}
}

void cu_tensor_layer_norm_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA> wt,
	const Tensor<float32, CUDA> bt,
	const float32 eps,
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int norm_region_size = wt.size();  // e.g. w shape: [4, 8] -> 32
	const int num_norm_regions = xt.size() / norm_region_size;  // e.g. x shape: [3, 4, 8] -> 3 

	auto* dx = xt.buffer();
	auto* dw = wt.buffer();
	auto* db = bt.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { 32, 8, 1 };
	dim3 gs = { 1, calc_req_num_blocks(num_norm_regions, bs.y), 1};
	
	cu_tensor_layer_norm_f32_kernel<<<gs, bs>>>(num_norm_regions, norm_region_size, dx, dw, db, eps, dy);
	CUDA_CHECK_LAST_ERROR();
}
