#include "transp_ops.cuh"

__global__ void tensor_transp_kernel_f32(const int m, const int n, const float32* dx, float32* dy)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tx < m && ty < n)
	{
		int offset_x = tx * n + ty;
		int offset_y = ty * m + tx;
		dy[offset_y] = dx[offset_x];
	}
}


void tensor_transp_f32(
	const Tensor<float32, CUDA>& x,
	const Tensor<float32, CUDA>& y)
{
	int m = x.shape[0];
	int n = x.shape[1];

	float32* dx = x.buffer();
	float32* dy = y.buffer();

	dim3 bs = { 16, 16, 1 };

	unsigned int gsx = m / bs.x + ((m % bs.x > 0) ? 1 : 0);  // vertical
	unsigned int gsy = n / bs.y + ((n % bs.y > 0) ? 1 : 0);  // horizontal
	dim3 gs = { gsx, gsy, 1 };

	tensor_transp_kernel_f32<<<gs, bs>>>(m, n, dx, dy);
	CUDA_CHECK_LAST_ERROR();
}




__global__ void tensor_transp_kernel_i8(const int m, const int n, const int8* dx, int8* dy)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tx < m && ty < n)
	{
		int offset_x = tx * n + ty;
		int offset_y = ty * m + tx;
		dy[offset_y] = dx[offset_x];
	}
}


void tensor_transp_i8(
	const Tensor<int8, CUDA>& x,
	const Tensor<int8, CUDA>& y)
{
	int m = x.shape[0];
	int n = x.shape[1];

	int8* dx = x.buffer();
	int8* dy = y.buffer();

	dim3 bs = { 16, 16, 1 };

	unsigned int gsx = m / bs.x + ((m % bs.x > 0) ? 1 : 0);  // vertical
	unsigned int gsy = n / bs.y + ((n % bs.y > 0) ? 1 : 0);  // horizontal
	dim3 gs = { gsx, gsy, 1 };

	tensor_transp_kernel_i8<<<gs, bs>>>(m, n, dx, dy);
	CUDA_CHECK_LAST_ERROR();
}
