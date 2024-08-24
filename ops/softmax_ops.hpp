#ifndef __SOFTMAX_OPS__
#define __SOFTMAX_OPS__

#include "softmax_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"

/*
*  Calculates the sofmtax for a tensor on CPU.
*  The reduced dimension is always the last one.
*  The input is assumed to be 2 dimensional.
*  (Otherwise it is recommended to use a reshape.)
*/
template<PreciseFloatType dtype>
Tensor<dtype, CPU> tensor_softmax(const Tensor<dtype, CPU>& x)
{
	assert(x->dim == 2);

	int m = x.shape[0];
	int n = x.shape[1];

	dtype* x_data = x.buffer();

	std::vector<int> y_shape({ m, n });
	Tensor<dtype, CPU> y(y_shape);
	dtype* y_data = y.buffer();

	// reference implementation
	// reliable (but slow)

	int x_stride_r = x.stride[0];
	int x_stride_c = x.stride[1];

	int y_stride_r = y.stride[0];
	int y_stride_c = y.stride[1];

	// iterate over all the rows and process a whole row
	for (int r = 0; r < m; ++r)
	{
		// caclulate the maximum of the row
		// the maximum is used to avoid numerical unstability
		int x_base_offset = r * x_stride_r;
		dtype max_val = x_data[x_base_offset];
		for (int c = 1; c < n; ++c)
		{
			dtype tentative_val = x_data[x_base_offset + c * x_stride_c];
			if (tentative_val > max_val)
			{
				max_val = tentative_val;
			}
		}

		// calculate the exponents and the sum reduce
		int y_base_offset = r * y_stride_r;

		// calculate for first element
		dtype exponent = x_data[x_base_offset] - max_val;
		dtype exp_val = static_cast<dtype>(std::exp(static_cast<double>(exponent)));
		y_data[y_base_offset] = exp_val;
		dtype reduced = exp_val;

		for (int c = 1; c < n; ++c)
		{
			exponent = x_data[x_base_offset + c * x_stride_c] - max_val;
			exp_val = static_cast<dtype>(std::exp(static_cast<double>(exponent)));
			y_data[y_base_offset + c * y_stride_c] = exp_val;
			reduced += exp_val;
		}

		// calculate the softmax by multiplication
		dtype rec_reduced = static_cast<dtype>(1.0) / reduced;

		for (int c = 0; c < n; ++c)
		{
			int offset = y_base_offset + c * y_stride_c;
			y_data[offset] = y_data[offset] * rec_reduced;
		}
	}

	return y;
}


template<typename dtype>
Tensor<dtype, CUDA> tensor_softmax(const Tensor<dtype, CUDA>& x)
{
	assert(x->dim == 2);

	int m = x.shape[0];
	int n = x.shape[1];
	std::vector<int> y_shape({ m, n });
	Tensor<dtype, CUDA> y(y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		tensor_softmax_f32(x, y);
	}
	else
	{
		static_assert(std::is_same_v<dtype, int32>, "Unsupported data type");
		//tensor_add_i32(kpms, lhs, rhs, res);
	}

	return y;
}

#endif  // __SOFTMAX_OPS__
