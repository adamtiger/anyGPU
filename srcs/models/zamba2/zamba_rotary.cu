#include "zamba_rotary.cuh"

/* precalculate zamba2 rotary embedding */

__global__ void cu_tensor_zamba_precomp_rotary_embedding_f32_kernel(
	const int32* p_data, // position ids data 
	const int32 num_pos_idcs,
	const int32 emb_size,
	const int32 base,
	float32* freq_data)  // output
{
	int32 i = blockDim.x * blockIdx.x + threadIdx.x;
	int32 s = blockDim.y * blockIdx.y + threadIdx.y;

	int32 d_per_2 = emb_size / 2;

	if (s < num_pos_idcs && i < d_per_2)
	{
		int32 m = p_data[s];
		int32 freq_offs = s * d_per_2 + i;
		freq_data[freq_offs] = (float32)m * powf((float32)base, -((float32)i / (float32)d_per_2));
	}
}


void cu_tensor_zamba_precomp_rotary_embedding_f32(
	const Tensor<int32, CUDA>& pt,
	const int32 emb_size,
	const int32 base,
	Tensor<float32, CUDA>& freq)
{
	int32 num_pos_idcs = pt.size();

	int32* p_data = pt.buffer();
	float32* freq_data = freq.buffer();

	dim3 bs = { 32, 4, 1 };
	dim3 gs = { calc_req_num_blocks(emb_size / 2, bs.x), calc_req_num_blocks(num_pos_idcs, bs.y), 1 };

	cu_tensor_zamba_precomp_rotary_embedding_f32_kernel<<<gs, bs>>>(p_data, num_pos_idcs, emb_size, base, freq_data);
	CUDA_CHECK_LAST_ERROR();
}


/* apply zamba2 rotary embedding */

__global__ void cu_tensor_apply_zamba_rotary_embedding_f32_kernel(
	const int32 batch,
	const int32 mid,
	const int32 emb_size,
	const int32 seq,
	const float32* dx,  // [batch, ..., seq, emb_size]
	const float32* df,  // [max_seq, emb_size/2]
	float32* dy)  // output
{
	int32 b = blockDim.z * blockIdx.z;
	int32 m = blockDim.y * blockIdx.y + threadIdx.y;
	int32 i = blockDim.x * blockIdx.x + threadIdx.x;

	if (b < batch && m < mid && i < emb_size / 2)
	{
		int32 s = m % seq;

		float32 angle = df[(b * seq + s) * emb_size / 2 + i];
		float32 cos_angle = cosf(angle);
		float32 sin_angle = sinf(angle);

		float32 x1 = dx[b * mid * emb_size + m * emb_size + i];
		float32 x2 = dx[b * mid * emb_size + m * emb_size + emb_size / 2 + i];

		float32 y1 = cos_angle * x1 - sin_angle * x2;
		float32 y2 = sin_angle * x1 + cos_angle * x2;

		dy[b * mid * emb_size + m * emb_size + i] = y1;
		dy[b * mid * emb_size + m * emb_size + emb_size / 2 + i] = y2;
	}
}


void cu_tensor_apply_zamba_rotary_embedding_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA>& ft,
	Tensor<float32, CUDA>& yt)
{
	// data buffers
	float32* x_data = xt.buffer();
	float32* f_data = ft.buffer();
	float32* y_data = yt.buffer();

	// sizes
	int32 batch = xt.shape[0];
	int32 emb_size = xt.shape[xt.dim - 1];
	int32 mid = xt.size() / (batch * emb_size);  // [..., seq] excluding the last dim
	int32 seq = xt.shape[xt.dim - 2];

	dim3 bs = { 32, 4, 1 };
	dim3 gs = {
		calc_req_num_blocks(emb_size / 2, bs.x),
		calc_req_num_blocks(mid, bs.y),
		calc_req_num_blocks(batch, 1)
	};

	cu_tensor_apply_zamba_rotary_embedding_f32_kernel<<<gs, bs>>>(
		batch, mid, emb_size, seq,
		x_data, f_data, y_data
	);
	CUDA_CHECK_LAST_ERROR();
}
