#include "embedding_ops.hpp"


__global__ void cu_tensor_embedding_f32_kernel(
	const int num_indices,
	const int embedding_size,
	const int32* x_data,
	const float32* w_data,  // embedding vectors
	float32* y_data)
{
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if (t < num_indices)
	{
		int32 i = x_data[t];
		for (int k = 0; k < embedding_size; ++k)  // TODO: consider using cumemcpy
		{
			y_data[t * embedding_size + k] = w_data[i * embedding_size + k];
		}
	}
}


void cu_tensor_embedding_f32(
	const Tensor<int32, CUDA>& xt,
	const Tensor<float32, CUDA>& wt,
	Tensor<float32, CUDA>& yt)
{
	const int num_indices = xt.size();
	const int emb_size = wt.shape[1];

	float32* y_data = yt.buffer();
	int32* x_data = xt.buffer();
	float32* w_data = wt.buffer();

	dim3 bs = {32 * 4, 1, 1};
	dim3 gs = {calc_req_num_blocks(num_indices, bs.x), 1, 1};

	cu_tensor_embedding_f32_kernel<<<gs, bs>>>(num_indices, emb_size, x_data, w_data, y_data);
	CUDA_CHECK_LAST_ERROR();
}
