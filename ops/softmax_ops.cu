#include "softmax_ops.cuh"

#include <limits>

constexpr int WARP_SIZE = 32;
constexpr float32 LOWEST_FLOAT32 = std::numeric_limits<float32>::lowest();

__global__ void tensor_softmax_kernel_f32(const int m, const int n, const float32* dx, float32* dy)
{
	int tx = threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	/*
	    The input is 2 dimensional (m, n).
		Each warp works on a complete row.
		A block can work on more than one 
		consecutive rows.
	*/

	if (ty < m)
	{
		// number of warp width region to reduce
		int num_segments = n / WARP_SIZE + (n % WARP_SIZE > 0 ? 1 : 0);

		// Mask for the threads.
		unsigned int mask = 0xffffffff;

		int row_offset = ty * n;


		// Find the maximum for a row

		//  First each thread calculates the maximum for all sub part,
		//  each sub part has warp size width
		float32 max_val = LOWEST_FLOAT32;
		for (int s = 0; s < num_segments; ++s)
		{
			int row_element_idx = s * WARP_SIZE + tx;
			if (row_element_idx < n)
			{
				float32 tentative_value = dx[row_offset + row_element_idx];
				max_val = fmaxf(max_val, tentative_value);
			}
		}

		//  Now each threads has its maximum from the corresponding places of
		//  the subparts.
		//  Calculate the maximum for a warp.
		for (int i = 16; i >= 1; i /= 2)
		{
			max_val = fmaxf(max_val, __shfl_xor_sync(mask, max_val, i, 32));
		}
			

		// Find the reduced value for a row.

		//  First each thread calculates the sum for all sub part,
		//  each sub part has warp size width
		float32 sum_val = 0.f;
		for (int s = 0; s < num_segments; ++s)
		{
			int row_element_idx = s * WARP_SIZE + tx;
			if (row_element_idx < n)
			{
				sum_val += expf(dx[row_offset + row_element_idx] - max_val);
			}
		}

		//  Now each threads has its sum from the corresponding places of
		//  the subparts.
		//  Calculate the sum for a warp.
		for (int i = 16; i >= 1; i /= 2)
		{
			sum_val += __shfl_xor_sync(mask, sum_val, i, 32);
		}


		// Calculate the softmax value for each element by multiplication.
		float32 rec_reduced = 1.f / sum_val;
		for (int s = 0; s < num_segments; ++s)
		{
			int row_element_idx = s * WARP_SIZE + tx;
			if (row_element_idx < n)
			{
				dy[row_offset + row_element_idx] = expf(dx[row_offset + row_element_idx] - max_val) * rec_reduced;
			}
		}
	}
}


void tensor_softmax_f32(
	const Tensor<float32, CUDA>& x,
	Tensor<float32, CUDA>& y)
{
	int m = x.shape[0];
	int n = x.shape[1];

	float32* dx = x.buffer();
	float32* dy = y.buffer();

	dim3 bs = { WARP_SIZE, 4, 1 };

	unsigned int gsx = 1;  // horizontal
	unsigned int gsy = m / bs.y + ((m % bs.y > 0) ? 1 : 0);  // vertical
	dim3 gs = { gsx, gsy, 1 };

	tensor_softmax_kernel_f32<<<gs, bs>>>(m, n, dx, dy);
}




__global__ void tensor_softmax_bwd_kernel_f32(
	const int m, const int n, 
	const float32* dx, 
	const float32* d_grad_y, 
	float32* d_grad_x)
{
	int tx = threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (ty < m)
	{
		// number of warp width region to reduce
		int num_segments = n / WARP_SIZE + (n % WARP_SIZE > 0 ? 1 : 0);

		int base_offset = ty * n;

		// calculate the grad_y * x (elementwise)
		float32 reduced = 0.f;
		for (int s = 0; s < num_segments; ++s)
		{
			int col_index = s * WARP_SIZE + tx;

			if (col_index < n)
			{
				int offset = base_offset + col_index;

				float32 temp = d_grad_y[offset] * dx[offset];
				reduced += temp;
				d_grad_x[offset] = temp;
			}			
		}

		// calculate the reduction over the warp
		for (int i = 16; i >= 1; i /= 2)
		{
			reduced += __shfl_xor_sync(0xffffffff, reduced, i, 32);
		}

		// calculate the gradient
		for (int s = 0; s < num_segments; ++s)
		{
			int col_index = s * WARP_SIZE + tx;

			if (col_index < n)
			{
				int offset = base_offset + col_index;

				d_grad_x[offset] -= dx[offset] * reduced;
			}
		}
	}
}


void tensor_softmax_bwd_f32(
	const Tensor<float32, CUDA>& x,
	const Tensor<float32, CUDA>& grad_y,
	Tensor<float32, CUDA>& grad_x)
{
	int m = x.shape[0];
	int n = x.shape[1];

	float32* dx = x.buffer();
	float32* gy_data = grad_y.buffer();
	float32* gx_data = grad_x.buffer();

	dim3 bs = { WARP_SIZE, 4, 1 };

	unsigned int gsx = 1;  // horizontal
	unsigned int gsy = m / bs.y + ((m % bs.y > 0) ? 1 : 0);  // vertical
	dim3 gs = { gsx, gsy, 1 };

	tensor_softmax_bwd_kernel_f32<<<gs, bs>>>(m, n, dx, gy_data, gx_data);
}
