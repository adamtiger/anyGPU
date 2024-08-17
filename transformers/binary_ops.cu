#include "binary_ops.cuh"

__global__ void tensor_add_kernel_f32(const int length, const float* dlhs, const float* drhs, float* dout)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < length)
	{
		dout[tid] = dlhs[tid] + drhs[tid];
	}
}

__global__ void tensor_add_kernel_i32(const int length, const int* dlhs, const int* drhs, int* dout)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < length)
	{
		dout[tid] = dlhs[tid] + drhs[tid];
	}
}

void tensor_add_f32(const dim3 gs, const dim3 bs, const int length, const float* dlhs, const float* drhs, float* dout)
{
	tensor_add_kernel_f32<<<gs, bs>>>(length, dlhs, drhs, dout);
}

void tensor_add_i32(const dim3 gs, const dim3 bs, const int length, const int* dlhs, const int* drhs, int* dout)
{
	tensor_add_kernel_i32<<<gs, bs>>>(length, dlhs, drhs, dout);
}

