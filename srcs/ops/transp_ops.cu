#include "transp_ops.cuh"

__global__ void cu_tensor_transp_kernel_f32(const int m, const int n, const float32* dx, float32* dy)
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


void cu_tensor_transp_f32(
	const Tensor<float32, CUDA>& x,
	Tensor<float32, CUDA>& y)
{
	int m = x.shape[0];
	int n = x.shape[1];

	float32* dx = x.buffer();
	float32* dy = y.buffer();

	dim3 bs = { 16, 16, 1 };

	unsigned int gsx = m / bs.x + ((m % bs.x > 0) ? 1 : 0);  // vertical
	unsigned int gsy = n / bs.y + ((n % bs.y > 0) ? 1 : 0);  // horizontal
	dim3 gs = { gsx, gsy, 1 };

	cu_tensor_transp_kernel_f32<<<gs, bs>>>(m, n, dx, dy);
	CUDA_CHECK_LAST_ERROR();
}




__global__ void cu_tensor_transp_kernel_i8(const int m, const int n, const int8* dx, int8* dy)
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


void cu_tensor_transp_i8(
	const Tensor<int8, CUDA>& x,
	Tensor<int8, CUDA>& y)
{
	int m = x.shape[0];
	int n = x.shape[1];

	int8* dx = x.buffer();
	int8* dy = y.buffer();

	dim3 bs = { 16, 16, 1 };

	unsigned int gsx = m / bs.x + ((m % bs.x > 0) ? 1 : 0);  // vertical
	unsigned int gsy = n / bs.y + ((n % bs.y > 0) ? 1 : 0);  // horizontal
	dim3 gs = { gsx, gsy, 1 };

	cu_tensor_transp_kernel_i8<<<gs, bs>>>(m, n, dx, dy);
	CUDA_CHECK_LAST_ERROR();
}




/* transpose with swapping two elements */

__constant__ int x_stride[MAX_TENSOR_DIM];
__constant__ int y_sw_stride[MAX_TENSOR_DIM];  // swapped at ax1, ax2 for easier calc.

__global__ void cu_tensor_transp_swap_f32_kernel(
	const int length, 
	const int dim,
	const float32* dx, 
	const int ax1, 
	const int ax2, 
	float32* dy)
{
	int x_offset = blockDim.x * blockIdx.x + threadIdx.x;

	if (x_offset < length)
	{
		int rm_offs = x_offset;
		int y_offset = 0;

		for (int ix = 0; ix < dim; ++ix)
		{
			int x_idx = rm_offs / x_stride[ix];
			rm_offs = rm_offs % x_stride[ix];

			y_offset += y_sw_stride[ix] * x_idx;
		}

		dy[y_offset] = dx[x_offset];
	}
}


void cu_tensor_transp_swap_f32(
	const Tensor<float32, CUDA>& x,
	const int32 ax1,
	const int32 ax2,
	Tensor<float32, CUDA>& y)
{
	int length = x.numel();
	int dim = x.dim;

	float32* dx = x.buffer();
	float32* dy = y.buffer();

	dim3 bs = { 32 * 4, 1, 1 };
	dim3 gs = { calc_req_num_blocks(length, bs.x), 1, 1 };

	Stride _y_sw_stride = y.stride;
	std::swap(_y_sw_stride[ax1], _y_sw_stride[ax2]);

	cudaMemcpyToSymbol(x_stride, x.stride.data(), sizeof(int) * MAX_TENSOR_DIM, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(y_sw_stride, _y_sw_stride.data(), sizeof(int) * MAX_TENSOR_DIM, 0, cudaMemcpyHostToDevice);

	cu_tensor_transp_swap_f32_kernel<<<gs, bs>>>(length, dim, dx, ax1, ax2, dy);
	CUDA_CHECK_LAST_ERROR();
}
