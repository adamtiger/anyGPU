#include "conv_ops.cuh"

__global__ void cu_tensor_conv2d_f32_v1_kernel(
	const float32* dx,
	const float32* dw,
	const float32* db,
	float32* dy,
	const int32 xheight,
	const int32 xwidth,
	const int32 nfeatures,
	const int32 oheight,
	const int32 owidth,
	const int32 stride_h,
	const int32 stride_w,
	const int32 pad_h,  // at start
	const int32 pad_w,
	const int32 kchannel,
	const int32 kheight,
	const int32 kwidth)
{
	int32 z = blockIdx.z;
	int32 b = z / nfeatures;
	int32 f = z % nfeatures;

	int32 oh = blockDim.y * blockIdx.y + threadIdx.y;
	int32 ow = blockDim.x * blockIdx.x + threadIdx.x;

	if (!(oh < oheight && ow < owidth))
		return;

	// calculate output value
	float32 conv_value = db[f];

	for (int32 kc = 0; kc < kchannel; ++kc)
	{
		int32 k_offset0 = f * kchannel * kheight * kwidth + kc * kheight * kwidth;
		int32 x_offset0 = b * kchannel * xheight * xwidth + kc * xheight * xwidth;

		for (int32 kh = 0; kh < kheight; ++kh)
		{
			for (int32 kw = 0; kw < kwidth; ++kw)
			{
				int32 i = oh * stride_h + kh - pad_h;
				int32 j = ow * stride_w + kw - pad_w;

				if ((0 <= i && i < xheight) && (0 <= j && j < xwidth))
				{
					int32 k_offset = k_offset0 + kh * kwidth + kw;
					int32 x_offset = x_offset0 + i * xwidth + j;
					conv_value += dx[x_offset] * dw[k_offset];
				}
			}
		}
	}

	dy[z * oheight * owidth + oh * owidth + ow] = conv_value;
}


void cu_tensor_conv2d_f32_v1(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA> wt,
	const Tensor<float32, CUDA> bt,
	const std::array<int32, 2>& stride,
	const std::array<int32, 4>& pads,
	Tensor<float32, CUDA>& yt)
{
	// get sizes
	int32 nbatch = xt.shape[0];
	int32 nfeatures = wt.shape[0];
	int32 oheight = yt.shape[2];
	int32 owidth = yt.shape[3];

	int32 xheight = xt.shape[2];
	int32 xwidth = xt.shape[3];

	int32 kchannel = wt.shape[1];
	int32 kheight = wt.shape[2];
	int32 kwidth = wt.shape[3];

	// data ptrs
	const float32* dx = xt.buffer();
	const float32* dw = wt.buffer();
	const float32* db = bt.buffer();
	float32* dy = yt.buffer();

	dim3 bs = { 32, 8, 1 };
	dim3 gs = { calc_req_num_blocks(owidth, bs.x), calc_req_num_blocks(oheight, bs.y), (unsigned int)(nbatch * nfeatures) };
	cu_tensor_conv2d_f32_v1_kernel<<<gs, bs>>>(
		dx, dw, db, dy, 
		xheight, xwidth,
		nfeatures, 
		oheight,
		owidth,
		stride[0],
		stride[1],
		pads[0],
		pads[1],
		kchannel,
		kheight,
		kwidth
	);
	CUDA_CHECK_LAST_ERROR();
}
