#include "gemma_update_mask.cuh"


__global__ void cu_tensor_gemma_slide_mask_f32_kernel(
	const float32* d_mask,
	const int32 sliding_window,
	const int32 batch,
	const int32 seq_len,
	const int32 mask_len,
	const float32 f32_min,
	float32* d_y)
{
	int tx = threadIdx.x;
	int seq_idx = blockIdx.x % seq_len;
	int batch_ix = blockIdx.x / seq_len;

	if (tx < mask_len && seq_idx < seq_len && batch_ix < batch)
	{
		int offset = batch_ix * seq_len * mask_len + seq_idx * mask_len + tx;

		if (seq_idx >= sliding_window && tx <= seq_idx - sliding_window)
		{
			d_y[offset] = f32_min;
		}
		else
		{
			d_y[offset] = d_mask[offset];
		}
	}
}


void cu_tensor_gemma_slide_mask_f32(
	const Tensor<float32, CUDA>& attention_mask,
	const int32 sliding_window,
	Tensor<float32, CUDA>& yt)
{
	const float32 f32_min = std::numeric_limits<float32>::lowest();

	// calculate sizes
	int32 batch = attention_mask.shape[0];
	int32 seq_len = attention_mask.shape[2];
	int32 mask_len = attention_mask.shape[3];

	// buffers
	const float32* d_mask = attention_mask.buffer();
	float32* d_y = yt.buffer();

	// launch parameters
	dim3 bs = { 32 * calc_req_num_blocks(mask_len, 32), 1, 1 };  // mask_len is quite small (41 or similar)
	dim3 gs = { (unsigned int)(batch * seq_len), 1, 1 };

	// calling the kernel
	cu_tensor_gemma_slide_mask_f32_kernel<<<gs, bs>>>(
		d_mask,
		sliding_window,
		batch, seq_len, mask_len,
		f32_min,
		d_y);
}
