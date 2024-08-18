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

void tensor_add_f32(
	const KernelParameters& kpms,
	const Tensor<float32, CUDA>& lhs,
	const Tensor<float32, CUDA>& rhs,
	const Tensor<float32, CUDA>& out)
{
	int length = lhs.size();
	float32* dlhs = lhs.buffer();
	float32* drhs = rhs.buffer();
	float32* dout = out.buffer();

	tensor_add_kernel_f32<<<kpms.grid_size, kpms.block_size>>>(length, dlhs, drhs, dout);
}

void tensor_add_i32(
	const KernelParameters& kpms,
	const Tensor<int32, CUDA>& lhs,
	const Tensor<int32, CUDA>& rhs,
	const Tensor<int32, CUDA>& out)
{
	int length = lhs.size();
	int32* dlhs = lhs.buffer();
	int32* drhs = rhs.buffer();
	int32* dout = out.buffer();

	tensor_add_kernel_i32<<<kpms.grid_size, kpms.block_size>>>(length, dlhs, drhs, dout);
}
