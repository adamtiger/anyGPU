#ifndef __TEST_TOOLS__
#define __TEST_TOOLS__

#include "tensor.hpp"
#include "core_concepts.hpp"

template<IntegerType T>
static bool compare_data_buffers(const Tensor<T, CPU>& actual, const Tensor<T, CPU>& expected)
{
	T* expected_data = expected.buffer();
	T* actual_data = actual.buffer();

	int length = expected.size();

	bool eq = true;
	for (int ix = 0; ix < length; ++ix)
	{
		eq = eq && (expected_data[ix] == actual_data[ix]);
	}
	return eq;
}

template<PreciseFloatType T>
static bool compare_data_buffers(const Tensor<T, CPU>& actual, const Tensor<T, CPU>& expected, const float32 abs_tol=0.001f)
{
	T* expected_data = expected.buffer();
	T* actual_data = actual.buffer();

	int length = expected.size();

	int cntr = 0;
	bool eq = true;
	for (int ix = 0; ix < length; ++ix)
	{
		eq = eq && std::abs(expected_data[ix] - actual_data[ix]) < (T)abs_tol;

		if (!eq && cntr++ < 20)
		{
			std::cout << ix << " [" << expected_data[ix] << " <> " << actual_data[ix] << "] \n";
		}
	}
	return eq;
}

template<PreciseFloatType T>
static bool compare_data_buffers_reldiff(const Tensor<T, CPU>& actual, const Tensor<T, CPU>& expected, const float32 rel_tol = 0.01f)
{
	T* expected_data = expected.buffer();
	T* actual_data = actual.buffer();

	int length = expected.size();

	int cntr = 0;
	bool eq = true;
	for (int ix = 0; ix < length; ++ix)
	{
		eq = eq && std::abs(expected_data[ix] - actual_data[ix]) / (std::abs(expected_data[ix]) + (T)1e-8) < (T)rel_tol;

		if (!eq && cntr++ < 10)
		{
			std::cout << ix << " [" << expected_data[ix] << " <> " << actual_data[ix] << "] \n";

		}
	}
	return eq;
}

/*
  The goal of this function is to compare the 
  tensors globally and fail if the difference is
  consistently large.
  It is used for float16 and other less accurate
  types. On cpu only float32 reference calculation
  is available.
*/
template<PreciseFloatType T>
static bool compare_data_buffers_l2(const Tensor<T, CPU>& actual, const Tensor<T, CPU>& expected, const float32 rel_l2 = 0.01f)
{
	T* expected_data = expected.buffer();
	T* actual_data = actual.buffer();

	int length = expected.size();

	float32 cum_square_diff = 0.f;
	float32 cum_square = 0.f;
	for (int ix = 0; ix < length; ++ix)
	{
		float32 temp = (float32)(expected_data[ix] - actual_data[ix]);
		float32 residual = temp * temp;
		cum_square_diff += residual;
		cum_square += (float32)(expected_data[ix]) * (float32)(expected_data[ix]) + 1e-8f;
	}

	float32 l2 = sqrtf(cum_square_diff / cum_square);

	bool eq = l2 < rel_l2;
	if (!eq)
	{
		std::cout << "L2 rel difference: " << l2 << std::endl;
	}

	return eq;
}

#endif  // __TEST_TOOLS__
