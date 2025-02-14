#ifndef __CONV_OPS__
#define __CONV_OPS__

#include "conv_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


/* convolutions */

static inline int32 _calc_conv_output_size_along_one_axis(
	const int32 input_size,
	const int32 kernel_size,
	const int32 stride,
	const int32 pad_start,  // for horizontal axis: left size
	const int32 pad_end)
{
	return (input_size + pad_start + pad_end - kernel_size) / stride + 1;
}


/*
  2D convolution.
  This convolution has no dilation and grouping.
  Only padding is allowed.
  Bias is compulsory.
  stride - sh, sw
  pads - ph_start, pw_start, ph_end, pw_end
*/
template<PreciseFloatType dtype>
inline Tensor<dtype, CPU> tensor_conv2d(
	const Tensor<dtype, CPU>& xt,      // batch, channel, height, width
	const Tensor<dtype, CPU> wt,       // num_features, channel, kernel_height, kernel_width
	const Tensor<dtype, CPU> bt,       // num_features
	const std::array<int32, 2>& stride,
	const std::array<int32, 4>& pads)
{
	// check input variables
	ACASSERT(xt.shape[1] == wt.shape[1], "kernel and input must have the same number of channels");
	ACASSERT(wt.shape[0] == bt.shape[0], "kernel feature num != bias size");

	// calculate output size
	int32 y_dim = 4;
	Shape y_shape;
	y_shape[0] = xt.shape[0];
	y_shape[1] = wt.shape[0];
	y_shape[2] = _calc_conv_output_size_along_one_axis(xt.shape[2], wt.shape[2], stride[0], pads[0], pads[2]);
	y_shape[3] = _calc_conv_output_size_along_one_axis(xt.shape[3], wt.shape[3], stride[1], pads[1], pads[3]);

	// create output tensor
	Tensor<dtype, CPU> yt(y_dim, y_shape);

	// calculate convolution (ref. impl.)
	const dtype* x_data = xt.buffer();
	const dtype* w_data = wt.buffer();
	const dtype* b_data = bt.buffer();
    dtype* y_data = yt.buffer();

	const int32 nbatch = yt.shape[0];
	const int32 nfeature = yt.shape[1];
	const int32 oheight = yt.shape[2];
	const int32 owidth = yt.shape[3];

	const int32 nchannel = xt.shape[1];
	const int32 xheight = xt.shape[2];
	const int32 xwidth = xt.shape[3];

	const int32 kheight = wt.shape[2];
	const int32 kwidth = wt.shape[3];


	for (int32 ob = 0; ob < nbatch; ++ob)
	{
		for (int32 oc = 0; oc < nfeature; ++oc)
		{
			for (int32 oh = 0; oh < oheight; ++oh)
			{
				for (int32 ow = 0; ow < owidth; ++ow)
				{
					// calculate the output value at (ob, oc, oh, ow)

					dtype conv_value = (dtype)0;

					int32 ib = ob;
					int32 kf = oc;

					for (int32 kc = 0; kc < nchannel; ++kc)
					{
						int32 x_offset0 = ib * xt.stride[0] + kc * xt.stride[1];
						int32 k_offset0 = kf * wt.stride[0] + kc * wt.stride[1];

						for (int32 kh = 0; kh < kheight; ++kh)
						{
							for (int32 kw = 0; kw < kwidth; ++kw)
							{
								int32 i = oh * stride[0] + kh - pads[0];
								int32 j = ow * stride[1] + kw - pads[1];

								if ((0 <= i && i < xheight) && (0 <= j && j < xwidth))  // 0 padded image, check if inside the original image
								{
									int32 x_offset = x_offset0 + i * xt.stride[2] + j * xt.stride[3];
									int32 k_offset = k_offset0 + kh * wt.stride[2] + kw * wt.stride[3];

									conv_value += x_data[x_offset] * w_data[k_offset];
								}
							}
						}
					}

					conv_value += b_data[oc];

					int32 y_offset = ob * yt.stride[0] + oc * yt.stride[1] + oh * yt.stride[2] + ow * yt.stride[3];
					y_data[y_offset] = conv_value;
				}
			}
		}
	}

	return yt;
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_conv2d(
	const Tensor<dtype, CUDA>& xt,
	const Tensor<dtype, CUDA> wt,
	const Tensor<dtype, CUDA> bt,
	const std::array<int32, 2>& stride,
	const std::array<int32, 4>& pads)
{
	// check input variables
	ACASSERT(xt.shape[1] == wt.shape[1], "kernel and input must have the same number of channels");
	ACASSERT(wt.shape[0] == bt.shape[0], "kernel feature num != bias size");

	// calculate output size
	int32 y_dim = 4;
	Shape y_shape;
	y_shape[0] = xt.shape[0];
	y_shape[1] = wt.shape[0];
	y_shape[2] = _calc_conv_output_size_along_one_axis(xt.shape[2], wt.shape[2], stride[0], pads[0], pads[2]);
	y_shape[3] = _calc_conv_output_size_along_one_axis(xt.shape[3], wt.shape[3], stride[1], pads[1], pads[3]);

	// create output tensor
	Tensor<dtype, CUDA> yt(y_dim, y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_conv2d_f32_v1(xt, wt, bt, stride, pads, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}


#endif  // __CONV_OPS__

