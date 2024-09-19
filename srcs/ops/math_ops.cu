#include "math_ops.cuh"

__global__ void cu_tensor_silu_f32_kernel(
	const int length,
	const float32* dx,
	float32* dy)
{
	// mapping the y block index and thread index to the parallel regions
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if (t < length)
	{
		float32 x = dx[t];
		dy[t] = x / (1.f + expf(-x));
	}
}

void cu_tensor_silu_f32(
	const Tensor<float32, CUDA>& xt,
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int length = xt.size();

	auto* dx = xt.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { 32 * 4, 1, 1 };
	dim3 gs = { calc_req_num_blocks(length, bs.x), 1, 1 };

	cu_tensor_silu_f32_kernel<<<gs, bs>>>(length, dx, dy);
	CUDA_CHECK_LAST_ERROR();
}
