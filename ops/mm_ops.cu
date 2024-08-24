#include "mm_ops.cuh"

__global__ void tensor_mm_kernel_f32(const int m, const int n, const int k, const float32* dlhs, const float32* drhs, float32* dout)
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


void tensor_mm_f32(
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

	tensor_mm_kernel_f32<<<gs, bs>>>(m, n, k, dlhs, drhs, dout);
}

