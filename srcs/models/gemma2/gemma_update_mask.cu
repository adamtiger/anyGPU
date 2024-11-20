#include "gemma_update_mask.cuh"


__global__ void cu_tensor_gemma_update_mask_f32_kernel(
	const float32* d_mask,
	const int32* d_pos,
	const int32 batch,
	const int32 mask_len,
	const int32 seq_len,
	const int32 trg_len,
	const float32 f32_min,
	float32* d_y)
{
	int tx = threadIdx.x;
	int seq_idx = blockIdx.x % seq_len;
	int batch_ix = blockIdx.x / seq_len;

	if (tx < trg_len && seq_idx < seq_len && batch_ix < batch)
	{
		bool is_zero = true;
		float32 value = 0.f;

		if (seq_idx < tx)
		{
			is_zero = false;
			value = f32_min;
		}

		if (d_pos[seq_idx] >= tx)
		{
			is_zero = true;
			value = 0.f;
		}

		if (tx < mask_len)
		{
			int32 mask = static_cast<int32>(d_mask[batch_ix * mask_len + tx]);

			if (mask != 0)
			{
				is_zero = false;  // assumes mask can be 0 or 1, but not -f32_min
			}

			if (is_zero)
			{
				value = f32_min;
			}
		}

		d_y[batch_ix * seq_len * trg_len + seq_idx * trg_len + tx] = value;
	}
}


void cu_tensor_gemma_update_mask_f32(
	const Tensor<float32, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& cache_position,
	const int32 trg_len,
	Tensor<float32, CUDA>& yt)
{
	const float32 f32_min = std::numeric_limits<float32>::lowest();

	// calculate sizes
	int32 batch = attention_mask.shape[0];
	int32 mask_len = attention_mask.shape[1];
	int32 seq_len = cache_position.shape[0];

	// buffers
	const float32* d_mask = attention_mask.buffer();
	const int32* d_pos = cache_position.buffer();
	float32* d_y = yt.buffer();

	// launch parameters
	dim3 bs = { 32 * calc_req_num_blocks(trg_len, 32), 1, 1 };  // trg_len is quite small (41 or similar)
	dim3 gs = { (unsigned int)(batch * seq_len), 1, 1 };

	// calling the kernel
	cu_tensor_gemma_update_mask_f32_kernel<<<gs, bs>>>(
		d_mask, d_pos, 
		batch, mask_len, seq_len, trg_len, 
		f32_min, 
		d_y
	);
}
