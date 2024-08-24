#include "tests.hpp"

#include "binary_ops.hpp"
#include "mm_ops.hpp"

void test_binary_add_f32()
{
	// cuda based calculation
	auto dta = crt_random_tensor<float32, CUDA>({ 5, 300 }, 11);
	auto dtb = crt_random_tensor<float32, CUDA>({ 5, 300 }, 18);
	auto dtc = tensor_add(dta, dtb);

	// cpu based calculation (expected result)
	auto hta = dta.copy_to_host();
	auto htb = dtb.copy_to_host();
	auto htc = tensor_add(hta, htb);

	// compare
	auto htc_from_cuda = dtc.copy_to_host();

	bool eq = elementwise_compatible(htc, htc_from_cuda);  // checks the sizes
	if (eq)
	{
		float32* expected = htc.buffer();
		float32* actual = htc_from_cuda.buffer();

		int length = htc.size();

		for (int ix = 0; ix < length; ++ix)
		{
			eq = eq && std::abs(expected[ix] - actual[ix]) < 0.001f;
		}
	}

	std::cout << "TestCase [test_binary_add_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}

void test_binary_add_i32()
{
	// cuda based calculation
	auto dta = crt_random_tensor<int32, CUDA>({ 5, 300 }, 11);
	auto dtb = crt_random_tensor<int32, CUDA>({ 5, 300 }, 18);
	auto dtc = tensor_add(dta, dtb);

	// cpu based calculation (expected result)
	auto hta = dta.copy_to_host();
	auto htb = dtb.copy_to_host();
	auto htc = tensor_add(hta, htb);

	// compare
	auto htc_from_cuda = dtc.copy_to_host();

	bool eq = elementwise_compatible(htc, htc_from_cuda);  // checks the sizes
	if (eq)
	{
		int32* expected = htc.buffer();
		int32* actual = htc_from_cuda.buffer();

		int length = htc.size();

		for (int ix = 0; ix < length; ++ix)
		{
			eq = eq && (expected[ix] == actual[ix]);
		}
	}

	std::cout << "TestCase [test_binary_add_i32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}

void test_binary_mul_f32()
{
	// cuda based calculation
	auto dta = crt_random_tensor<float32, CUDA>({ 5, 300 }, 11);
	float32 dtb = 0.5f;
	auto dtc = tensor_mul(dta, dtb);

	// cpu based calculation (expected result)
	auto hta = dta.copy_to_host();
	auto htb = dtb;  // only scalar
	auto htc = tensor_mul(hta, htb);

	// compare
	auto htc_from_cuda = dtc.copy_to_host();

	bool eq = elementwise_compatible(htc, htc_from_cuda);  // checks the sizes
	if (eq)
	{
		float32* expected = htc.buffer();
		float32* actual = htc_from_cuda.buffer();

		int length = htc.size();

		for (int ix = 0; ix < length; ++ix)
		{
			eq = eq && std::abs(expected[ix] - actual[ix]) < 0.001f;
		}
	}

	std::cout << "TestCase [test_binary_mul_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}

void test_binary_mul_i32()
{
	// cuda based calculation
	auto dta = crt_random_tensor<int32, CUDA>({ 5, 300 }, 11);
	int32 dtb = 2;
	auto dtc = tensor_mul(dta, dtb);

	// cpu based calculation (expected result)
	auto hta = dta.copy_to_host();
	auto htb = dtb;
	auto htc = tensor_mul(hta, htb);

	// compare
	auto htc_from_cuda = dtc.copy_to_host();

	bool eq = elementwise_compatible(htc, htc_from_cuda);  // checks the sizes
	if (eq)
	{
		int32* expected = htc.buffer();
		int32* actual = htc_from_cuda.buffer();

		int length = htc.size();

		for (int ix = 0; ix < length; ++ix)
		{
			eq = eq && (expected[ix] == actual[ix]);
		}
	}

	std::cout << "TestCase [test_binary_mul_i32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_mm_f32()
{
	// cuda based calculation
	auto dta = crt_random_tensor<float32, CUDA>({ 40, 30 }, 11);
	auto dtb = crt_random_tensor<float32, CUDA>({ 30, 50 }, 18);
	auto dtc = tensor_mm(dta, dtb);

	// cpu based calculation (expected result)
	auto hta = dta.copy_to_host();
	auto htb = dtb.copy_to_host();
	auto htc = tensor_mm(hta, htb);

	// compare
	auto htc_from_cuda = dtc.copy_to_host();

	bool eq = elementwise_compatible(htc, htc_from_cuda);  // checks the sizes
	if (eq)
	{
		float32* expected = htc.buffer();
		float32* actual = htc_from_cuda.buffer();

		int length = htc.size();

		for (int ix = 0; ix < length; ++ix)
		{
			eq = eq && std::abs(expected[ix] - actual[ix]) < 0.001f;
		}
	}

	std::cout << "TestCase [test_mm_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}
