#include "quantize_ops.cuh"

constexpr int32 INT8_LOWEST = std::numeric_limits<int8>::lowest();
constexpr int32 INT8_HIGHEST = std::numeric_limits<int8>::max();


__global__ void tensor_quantize_linear_cuda_f32_i8_kernel(
	const int length, 
	const float32* dx, 
	const float32 scale, 
	const int8 zp, 
	int8* dy)
{
	int tix = blockDim.x * blockIdx.x + threadIdx.x;

	if (tix < length)
	{
		 int32 qx = (int32)lroundf(dx[tix] / scale) + (int32)zp;  // halfway cases: round away from 0
		 qx = min(max(qx, INT8_LOWEST), INT8_HIGHEST);
		 dy[tix] = (int8)qx;
	}
}

void tensor_quantize_linear_cuda_f32_i8(
	const Tensor<float32, CUDA>& x,
	const float32 scale,
	const int8 bias,
	Tensor<int8, CUDA>& y)
{
	int length = x.size();
	float32* dx = x.buffer();
	int8* dy = y.buffer();

	// grid dimensions
	dim3 bs = { 256, 1, 1 };

	dim3 gs = { calc_req_num_blocks(length, bs.x), 1, 1 };

	tensor_quantize_linear_cuda_f32_i8_kernel<<<gs, bs>>>(length, dx, scale, bias, dy);
}


__global__ void tensor_dequantize_linear_cuda_i8_f32_kernel(
	const int length,
	const int8* dx,
	const float32 scale,
	const int8 zp,
	float32* dy)
{
	int tix = blockDim.x * blockIdx.x + threadIdx.x;

	if (tix < length)
	{
		float32 qx = (float32)((int32)dx[tix] - (int32)zp) * scale;
		dy[tix] = qx;
	}
}

void tensor_dequantize_linear_cuda_i8_f32(
	const Tensor<int8, CUDA>& x,
	const float32 scale,
	const int8 bias,
	Tensor<float32, CUDA>& y)
{
	int length = x.size();
	int8* dx = x.buffer();
	float32* dy = y.buffer();

	// grid dimensions
	dim3 bs = { 256, 1, 1 };

	dim3 gs = { calc_req_num_blocks(length, bs.x), 1, 1 };

	tensor_dequantize_linear_cuda_i8_f32_kernel<<<gs, bs>>>(length, dx, scale, bias, dy);
}


__global__ void tensor_qmm_cuda_i8_f32_kernel(
	const int m, const int n, const int k,
	const int8* da, const int8* db,
	const float32 sa, const int8 zpa,
	const float32 sb, const int8 zpb,
	const float32 sy, const int8 zpy,
	int8* dy)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tx < n && ty < m)
	{
		float32 s = (sa * sb) / sy;

		int32 accumulator = 0;
		for (int c = 0; c < k; ++c)
		{
			int offs_a = ty * k + c;
			int offs_b = c * n + tx;
			accumulator += ((int16)da[offs_a] - (int16)zpa) * ((int16)db[offs_b] - (int16)zpb);
		}

		int32 qx = (int32)lroundf((float32)accumulator * s) + (int32)zpy;
		qx = min(max(qx, INT8_LOWEST), INT8_HIGHEST);
		dy[ty * n + tx] = (int8)qx;
	}
}

void tensor_qmm_cuda_i8_f32(
	const Tensor<int8, CUDA>& a,
	const Tensor<int8, CUDA>& b,
	const float32 sa, const int8 zpa,
	const float32 sb, const int8 zpb,
	const float32 sy, const int8 zpy,
	Tensor<int8, CUDA>& y)
{
	int m = a.shape[0];
	int n = b.shape[1];
	int k = a.shape[1];

	int8* da = a.buffer();
	int8* db = b.buffer();
	int8* dy = y.buffer();

	dim3 bs = { 32, 8, 1 };
	dim3 gs = { calc_req_num_blocks(n, bs.x), calc_req_num_blocks(m, bs.y), 1 };

	tensor_qmm_cuda_i8_f32_kernel<<<gs, bs>>>(m, n, k, da, db, sa, zpa, sb, zpb, sy, zpy, dy);
}
