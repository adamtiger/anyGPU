#include "math_ops.cuh"

__global__ void cu_tensor_zamba_glu_f32_kernel(
	const int n,
	const int h,
	const float32* dx,
	float32* dy)
{
	int i = blockIdx.x;
	int t = threadIdx.x;
	int w = blockDim.x;  // width of a threadblock

	if (i < n)
	{
		int s = h / w + (h % w > 0 ? 1 : 0);

		for (int sx = 0; sx < s; ++sx)
		{
			int j = sx * w + t;
			float32 x0 = dx[i * h * 2 + j];
			float32 gelu_x = 0.5f * x0 * (1.f + erff(x0 / sqrtf(2.f)));
			float32 x1 = dx[i * h * 2 + h + j];

			dy[i * h + j] = gelu_x * x1;
		}
	}
}

void cu_tensor_zamba_glu_f32(
	const Tensor<float32, CUDA>& xt,
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

	cu_tensor_zamba_glu_f32_kernel<<<gs, bs >>>(n, h, dx, dy);
	CUDA_CHECK_LAST_ERROR();
}
