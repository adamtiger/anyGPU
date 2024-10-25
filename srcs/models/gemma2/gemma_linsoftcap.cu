#include "gemma_linsoftcap.cuh"

__device__ float32 softcap(const float32 x, const float32 final_softcapping)
{
	auto temp = x / final_softcapping;
	temp = tanhf(temp);
	temp = temp * final_softcapping;
	return temp;
}

__global__ void cu_tensor_gemma_linear_softcap_f32_kernel(
	const int n,        // batch x seq_len
	const int hsize,    // hidden_size
	const int vsize,    // vocab_size
	const float32* dx,  // [n, hidden_size]
	const float32* dw,  // [vocab_size, hidden_size]
	const float32 final_softcapping,
	float32* dy)  // [n, vocab_size]
{
	int v = blockDim.x * blockIdx.x + threadIdx.x;
	int s = blockDim.y * blockIdx.y + threadIdx.y;

	if (s < n && v < vsize)
	{
		float32 acc = 0.f;
		for (int h = 0; h < hsize; ++h)
		{
			acc += dx[s * hsize + h] * dw[v * hsize + h];
		}

		dy[s * vsize + v] = softcap(acc, final_softcapping);
	}
}

void cu_tensor_gemma_linear_softcap_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA>& wt,
	const float32 final_softcapping,
	Tensor<float32, CUDA>& yt)
{
	const int n = xt.shape[0] * xt.shape[1];
	const int hsize = wt.shape[1];
	const int vsize = wt.shape[0];

	const float32* dx = xt.buffer();
	const float32* dw = wt.buffer();
	float32* dy = yt.buffer();

	dim3 bs = { 32, 8, 1 };
	dim3 gs = { calc_req_num_blocks(vsize, bs.x), calc_req_num_blocks(n, bs.y), 1 };

	cu_tensor_gemma_linear_softcap_f32_kernel<<<gs, bs>>>(n, hsize, vsize, dx, dw, final_softcapping, dy);
}
