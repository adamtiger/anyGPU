#include "causal_conv1d.cuh"

constexpr int WARP_SIZE = 32;
constexpr int WARP_NUM = 4;  // number of warps in a block
constexpr int MAX_KWIDTH = 4;

__global__ void cu_tensor_causal_conv1d_f32_kernel(
	const int batch,
	const int dims,
	const int slen,
	const int kwidth,
	const float32* dx,
	const float32* dw,
	const float32* db,
	float32* dy)
{
	int tx = threadIdx.x;  // a warp sweeps through the whole sequence
	int ty = blockDim.y * blockIdx.y + threadIdx.y;  // blocks are assigned to group of rows

	if (tx < slen && ty < batch * dims)
	{
		// read the corresponding kernel weights into registers
		float32 kweights[MAX_KWIDTH + 1] = {};
		int g = ty % dims;
		for (int k = 0; k < kwidth; ++k)
		{
			kweights[k] = dw[g * kwidth + k];
		}
		kweights[MAX_KWIDTH] = db[g];  // bias

		// iterate over the current sequence (row)
		__shared__ float32 segment_data[WARP_NUM][(WARP_SIZE + MAX_KWIDTH - 1)];

		// init the padding with zeros at back part of the segment,
		// back because of the copy from back to front (see below)
		for (int i = 0; i < kwidth - 1; ++i)  
			segment_data[threadIdx.y][WARP_SIZE + i] = 0.f;

		int num_segments = slen / WARP_SIZE + (slen % WARP_SIZE > 0 ? 1 : 0);
		for (int s = 0; s < num_segments; ++s)
		{
			const int glob_offset = ty * slen + s * WARP_SIZE;
			const int sh_offset = kwidth - 1;

			// moves the back kwidth - 1 pieces of elements to the front
			// this avoids to use a fixed size shared mem. with upper size limit (generality)
			// TODO: efficiency should be analysed
			for (int i = 0; i < kwidth - 1; ++i)
			{
				segment_data[threadIdx.y][i] = segment_data[threadIdx.y][WARP_SIZE + i];
			}

			// reads the current values into shared memory
			if (s * WARP_SIZE + tx < slen)  // checks if inside sequence (row)
			{
				
				segment_data[threadIdx.y][sh_offset + tx] = dx[glob_offset + tx];
			}

			__syncwarp();  // because convolution uses neighboring elements from shared mem.

			if (s * WARP_SIZE + tx < slen)  // checks if inside sequence (row)
			{
				// calculate the convolution
				float32 out = kweights[MAX_KWIDTH];
				for (int k = 0; k < kwidth; ++k)
				{
					out += kweights[k] * segment_data[threadIdx.y][tx + k];
				}

				// store in output
				dy[glob_offset + tx] = out / (1.f + expf(-out));;
			}
		}
	}
}

void cu_tensor_causal_conv1d_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA>& wt,
	const Tensor<float32, CUDA>& bt,
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int batch = xt.shape[0];
	const int dims = xt.shape[1];
	const int slen = xt.shape[2];
	const int kwidth = wt.shape[1];

	auto* dx = xt.buffer();
	auto* dw = wt.buffer();
	auto* db = bt.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { WARP_SIZE, 4, 1 };
	dim3 gs = { 1, calc_req_num_blocks(dims * batch, bs.y), 1};

	cu_tensor_causal_conv1d_f32_kernel<<<gs, bs>>>(batch, dims, slen, kwidth, dx, dw, db, dy);
	CUDA_CHECK_LAST_ERROR();
}
