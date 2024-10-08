#include "causal_conv1d.cuh"

__global__ void cu_tensor_causal_conv1d_f32_kernel(
	const int n,
	const int h,
	const float32* dx,
	float32* dy)
{
	// TODO: impl.
}

void cu_tensor_causal_conv1d_f32(  // TODO: impl.
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA>& wt,
	const Tensor<float32, CUDA>& bt,
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int h = xt.shape[xt.dim - 1] / 2;  // last axis size, hidden dim size
	const int n = yt.size() / h;  // number of slices, (number of elements without last axis)

	auto* dx = xt.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { 32 * 4, 1, 1 };
	dim3 gs = { (unsigned int)n, 1, 1 };

	//cu_tensor_causal_conv1d_f32_kernel<<<gs, bs>>>(n, h, dx, dy);
	CUDA_CHECK_LAST_ERROR();
}
