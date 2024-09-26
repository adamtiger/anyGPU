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


/* precalculate rotary embedding */

__global__ void cu_tensor_precomp_rotary_embedding_f32_kernel(
	const int32 max_seq_len,
	const int32 emb_size,
	const int32 base, 
	float32* freq_data)  // output
{
	int32 i = blockDim.x * blockIdx.x + threadIdx.x;
	int32 m = blockDim.y * blockIdx.y + threadIdx.y;

	int32 d_per_2 = emb_size / 2;

	if (m < max_seq_len && i < d_per_2)
	{
		int32 freq_offs = m * d_per_2 + i;
		freq_data[freq_offs] = (float32)m * powf((float32)base, -((float32)i / (float32)d_per_2));
	}
}


void cu_tensor_precomp_rotary_embedding_f32(
	const int32 max_seq_len,
	const int32 emb_size,
	const int32 base,
	Tensor<float32, CUDA>& freq)
{
	float32* freq_data = freq.buffer();

	dim3 bs = { 32, 4, 1 };
	dim3 gs = { calc_req_num_blocks(emb_size / 2, bs.x), calc_req_num_blocks(max_seq_len, bs.y), 1};

	cu_tensor_precomp_rotary_embedding_f32_kernel<<<gs, bs>>>(max_seq_len, emb_size, base, freq_data);
	CUDA_CHECK_LAST_ERROR();
}


/* apply rotary embedding */

__global__ void cu_tensor_apply_rotary_embedding_f32_kernel(
	const int32 batch,
	const int32 mid,
	const int32 emb_size,
	const int32 seq_stride,
	const float32* dx,  // [batch, seq, ..., emb_size]
	const int32* dp,    // [batch, seq]
	const float32* df,  // [max_seq, emb_size/2]
	float32* dy)  // output
{
	int32 b = blockDim.z * blockIdx.z;
	int32 m = blockDim.y * blockIdx.y + threadIdx.y;
	int32 i = blockDim.x * blockIdx.x + threadIdx.x;

	if (b < batch && m < mid && i < emb_size / 2)
	{
		int32 seq = mid * emb_size / seq_stride;
		int32 s = m * emb_size / seq_stride;

		int32 pos_idx = dp[b * seq + s];

		float32 angle = df[pos_idx * emb_size / 2 + i];
		float32 cos_angle = cosf(angle);
		float32 sin_angle = sinf(angle);

		float32 x1 = dx[b * mid * emb_size + m * emb_size + i * 2];
		float32 x2 = dx[b * mid * emb_size + m * emb_size + i * 2 + 1];

		float32 y1 = cos_angle * x1 - sin_angle * x2;
		float32 y2 = sin_angle * x1 + cos_angle * x2;

		dy[b * mid * emb_size + m * emb_size + i * 2] = y1;
		dy[b * mid * emb_size + m * emb_size + i * 2 + 1] = y2;
	}
}


void cu_tensor_apply_rotary_embedding_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<int32, CUDA>& pt,
	const Tensor<float32, CUDA>& ft,
	Tensor<float32, CUDA>& yt)
{
	// data buffers
	float32* x_data = xt.buffer();
	int32* p_data = pt.buffer();
	float32* f_data = ft.buffer();
	float32* y_data = yt.buffer();

	// sizes
	int32 batch = xt.shape[0];
	int32 emb_size = xt.shape[xt.dim - 1];
	int32 mid = xt.size() / (batch * emb_size);  // [seq, ...] excluding the last dim
	int32 seq_stride = xt.stride[1];

	dim3 bs = { 32, 4, 1 };
	dim3 gs = { 
		calc_req_num_blocks(emb_size / 2, bs.x), 
		calc_req_num_blocks(mid, bs.y), 
		calc_req_num_blocks(batch, 1)
	};

	cu_tensor_apply_rotary_embedding_f32_kernel<<<gs, bs>>>(
		batch, mid, emb_size, seq_stride,
		x_data, p_data, 
		f_data, y_data
	);
	CUDA_CHECK_LAST_ERROR();
}
