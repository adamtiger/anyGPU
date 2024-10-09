#ifndef __CAUSAL_CONV1D__
#define __CAUSAL_CONV1D__

#include "causal_conv1d.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"

/*
  The causal conv1d is a special variant
  of conv1d and it is fused with and 
  activation function too. 
  For zamba2 it is silu.

  Shape info:
      x shape: (batch, dims, seq_len)
      weights shape: (dims, width)
      bias shape: (dims)
      y shape: (batch, dims, seq_len)
  
  Implementation info:
      silu(conv1d(x, weights.unsqueeze(1), bias, stride=1, padding=width-1, groups=dims))[..., seq_len]
*/
template<PreciseFloatType dtype>
static Tensor<dtype, CPU> tensor_causal_conv1d(
	const Tensor<dtype, CPU>& xt, 
	const Tensor<dtype, CPU>& wt, 
	const Tensor<dtype, CPU>& bt)
{
	// checking input tensors sizes
	ACASSERT(xt.dim == 3, "xt needs to be 3 dimensional");
	ACASSERT(xt.shape[1] == wt.shape[0], "xt and wt need to have the same dims");
	ACASSERT(wt.shape[0] == bt.shape[0], "wt and bt need to have the same dims");

	// access the data arrays
	int y_dim = xt.dim;
	Shape y_shape = xt.shape;
	Tensor<dtype, CPU> yt(y_dim, y_shape);
	dtype* y_data = yt.buffer();
	dtype* x_data = xt.buffer();
	dtype* w_data = wt.buffer();
	dtype* b_data = bt.buffer();

	// reference implementation
	// reliable (but slow)
	int b = y_shape[0];
	int d = y_shape[1];
	int s = y_shape[2];
	int w = wt.shape[1];

	int x_offset = 0;
	int y_offset = 0;
	int w_offset = 0;

	for (int bix = 0; bix < b; ++bix)
	{
		for (int dix = 0; dix < d; ++dix)
		{
			x_offset = bix * xt.stride[0] + dix * xt.stride[1];
			y_offset = bix * yt.stride[0] + dix * yt.stride[1];
			w_offset = dix * wt.stride[0];

			// convolution over the 1d sequence
			for (int six = 0; six < s; ++six)
			{
				dtype out = 0;
				for (int k = 0; k < w; ++k)
				{
					int delta = k + 1 + six - w;
					if (delta >= 0)  // otherwise it is the 0 padding
					{
						out += w_data[w_offset + k] * x_data[x_offset + delta];
					}
				}

				out += b_data[dix];

				// silu activation
				out = out / (static_cast<dtype>(1.0) + exp(-out));

				y_data[y_offset + six] = out;
			}
		}
	}

	return yt;
}


template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_causal_conv1d(
	const Tensor<dtype, CUDA>& xt,
	const Tensor<dtype, CUDA>& wt,
	const Tensor<dtype, CUDA>& bt)
{
	// checking input tensors sizes
	ACASSERT(xt.dim == 3, "xt needs to be 3 dimensional");
	ACASSERT(xt.shape[1] == wt.shape[0], "xt and wt need to have the same dims");
	ACASSERT(wt.shape[0] == bt.shape[0], "wt and bt need to have the same dims");
	ACASSERT(wt.shape[1] <= 4, "kernel width needs to be under 4");

	// access the data arrays
	int y_dim = xt.dim;
	Shape y_shape = xt.shape;
	Tensor<dtype, CUDA> yt(y_dim, y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_causal_conv1d_f32(xt, wt, bt, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}

#endif  // __CAUSAL_CONV1D__
