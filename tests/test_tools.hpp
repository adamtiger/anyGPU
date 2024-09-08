#ifndef __TEST_TOOLS__
#define __TEST_TOOLS__

#include "tensor.hpp"
#include "core_concepts.hpp"

template<typename T>
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
static bool compare_data_buffers(const Tensor<T, CPU>& actual, const Tensor<T, CPU>& expected)
{
	T* expected_data = expected.buffer();
	T* actual_data = actual.buffer();

	int length = expected.size();

	bool eq = true;
	for (int ix = 0; ix < length; ++ix)
	{
		eq = eq && std::abs(expected_data[ix] - actual_data[ix]) < (T)0.001f;
	}
	return eq;
}


#endif  // __TEST_TOOLS__
