#include "tests.hpp"

#include "tensor.hpp"
#include "ops.hpp"
#include "attention.hpp"


void test_binary_add_f32()
{
	// cuda based calculation
	auto dta = crt_random_tensor<float32, CUDA>({ 5, 300 }, 11);
	auto dtb = crt_ones_tensor<float32, CUDA>({5, 300}); //crt_random_tensor<float32, CUDA>({ 5, 300 }, 18);
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


void test_transp_f32()
{
	// cuda based calculation
	auto dta = crt_random_tensor<float32, CUDA>({ 40, 30 }, 11);
	auto dtc = tensor_transp(dta);

	// cpu based calculation (expected result)
	auto hta = dta.copy_to_host();
	auto htc = tensor_transp(hta);

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

	std::cout << "TestCase [test_transp_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_softmax_f32()
{
	// cuda based calculation
	auto dta = crt_random_tensor<float32, CUDA>({ 6, 45 }, 11);
	auto dtc = tensor_softmax(dta);

	// cpu based calculation (expected result)
	auto hta = dta.copy_to_host();
	auto htc = tensor_softmax(hta);

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

			if (!eq)
			{
				std::cout << expected[ix] << " " << actual[ix] << "\n";
			}
		}
	}

	std::cout << "TestCase [test_softmax_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_softmax_bwd_f32()
{
	// cuda based calculation
	auto dtx = crt_random_tensor<float32, CUDA>({ 16, 64 }, 11);
	auto dtgy = crt_random_tensor<float32, CUDA>({ 16, 64 }, 14);  // gradient
	auto dtc = tensor_softmax_bwd(dtx, dtgy);

	// cpu based calculation (expected result)
	auto htx = dtx.copy_to_host();
	auto htgy = dtgy.copy_to_host();
	auto htc = tensor_softmax_bwd(htx, htgy);

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

			if (!eq)
			{
				std::cout << expected[ix] << " " << actual[ix] << "\n";
			}
		}
	}

	std::cout << "TestCase [test_softmax_bwd_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_sdp_fwd_f32()
{
	// cuda based calculation
	auto dqw = crt_random_tensor<float32, CUDA>({ 14, 64 }, 11);
	auto dkw = crt_random_tensor<float32, CUDA>({ 14, 64 }, 18);
	auto dvw = crt_random_tensor<float32, CUDA>({ 14, 64 }, 15);
	auto dy = single_head_attention_fwd<float32, CUDA, NONE, FULL>(dqw, dkw, dvw);

	// cpu based calculation (expected result)
	auto hqw = dqw.copy_to_host();
	auto hkw = dkw.copy_to_host();
	auto hvw = dvw.copy_to_host();
	auto hy = single_head_attention_fwd<float32, CPU, NONE, FULL>(hqw, hkw, hvw);

	// compare
	auto hy_from_cuda = dy.copy_to_host();

	bool eq = elementwise_compatible(hy, hy_from_cuda);  // checks the sizes
	if (eq)
	{
		float32* expected = hy.buffer();
		float32* actual = hy_from_cuda.buffer();

		int length = hy.size();

		for (int ix = 0; ix < length; ++ix)
		{
			eq = eq && std::abs(expected[ix] - actual[ix]) < 0.001f;

			if (!eq)
			{
				std::cout << expected[ix] << " " << actual[ix] << "\n";
			}
		}
	}

	std::cout << "TestCase [test_sdp_fwd_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_sdp_bwd_f32()
{
	// cuda based calculation
	auto dqw = crt_random_tensor<float32, CUDA>({ 14, 64 }, 11);
	auto dkw = crt_random_tensor<float32, CUDA>({ 14, 64 }, 18);
	auto dvw = crt_random_tensor<float32, CUDA>({ 14, 64 }, 15);
	auto dgy = crt_random_tensor<float32, CUDA>({ 14, 64 }, 10);
	SDPGradient dgrads = single_head_attention_bwd<float32, CUDA, NONE, FULL>(dqw, dkw, dvw, dgy);

	// cpu based calculation (expected result)
	auto hqw = dqw.copy_to_host();
	auto hkw = dkw.copy_to_host();
	auto hvw = dvw.copy_to_host();
	auto hgy = dgy.copy_to_host();
	SDPGradient hgrads = single_head_attention_bwd<float32, CPU, NONE, FULL>(hqw, hkw, hvw, hgy);

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
	{
		bool eq = elementwise_compatible(expected, actual);  // checks the sizes
		if (eq)
		{
			float32* ex = expected.buffer();
			float32* ac = actual.buffer();

			int length = expected.size();

			for (int ix = 0; ix < length; ++ix)
			{
				eq = eq && std::abs(ex[ix] - ac[ix]) < 0.001f;

				if (!eq)
				{
					std::cout << ex[ix] << " " << ac[ix] << "\n";
				}
			}
		}

		return eq;
	};

	bool eq = true;

	auto h_grad_q_from_cuda = dgrads.grad_q.copy_to_host();
	eq = eq && cmp(hgrads.grad_q, h_grad_q_from_cuda);

	auto h_grad_k_from_cuda = dgrads.grad_k.copy_to_host();
	eq = eq && cmp(hgrads.grad_k, h_grad_k_from_cuda);

	auto h_grad_v_from_cuda = dgrads.grad_v.copy_to_host();
	eq = eq && cmp(hgrads.grad_v, h_grad_v_from_cuda);

	std::cout << "TestCase [test_sdp_bwd_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_quant_sdp_fwd_f32_i8()
{
	// cuda based calculation
	auto dqw = crt_random_tensor<int8, CUDA>({ 21, 64 }, 11);
	auto dkw = crt_random_tensor<int8, CUDA>({ 21, 64 }, 28);
	auto dvw = crt_random_tensor<int8, CUDA>({ 21, 64 }, 45);

	float32 sq = 2.1f;
	int8 zpq = 16;

	float32 sk = 1.1f;
	int8 zpk = 24;

	float32 sv = 1.7f;
	int8 zpv = -5;

	float32 s1 = 2.3f;
	int8 zp1 = 17;

	float32 s2 = 1.5f;
	int8 zp2 = 22;

	float32 s3 = 1.4f;
	int8 zp3 = 5;

	float32 sy = 1.3f;
	int8 zpy = 3;

	auto dy = quantized_single_head_attention_fwd<float32, int8, CUDA, NONE, FULL>(
		dqw, dkw, dvw,
		sq, zpq, sk, zpk, sv, zpv,
		s1, zp1, s2, zp2, s3, zp3,
		sy, zpy
	);

	// cpu based calculation (expected result)
	auto hqw = dqw.copy_to_host();
	auto hkw = dkw.copy_to_host();
	auto hvw = dvw.copy_to_host();
	auto hy = quantized_single_head_attention_fwd<float32, int8, CPU, NONE, FULL>(
		hqw, hkw, hvw,
		sq, zpq, sk, zpk, sv, zpv,
		s1, zp1, s2, zp2, s3, zp3,
		sy, zpy
	);

	// compare
	auto hy_from_cuda = dy.copy_to_host();

	bool eq = elementwise_compatible(hy, hy_from_cuda);  // checks the sizes
	if (eq)
	{
		int8* expected = hy.buffer();
		int8* actual = hy_from_cuda.buffer();

		int length = hy.size();

		for (int ix = 0; ix < length; ++ix)
		{
			eq = eq && (expected[ix] == actual[ix]);

			if (!eq)
			{
				std::cout << (int32)expected[ix] << " " << (int32)actual[ix] << "\n";
			}
		}
	}

	std::cout << "TestCase [test_quant_sdp_fwd_f32_i8]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_quant_lin_f32_i8()
{
	// cuda based calculation
	auto dx = crt_random_tensor<float32, CUDA>({ 18, 64 }, 21);
	float32 scale = 2.f;
	int8 bias = 24;

	auto dy = tensor_quantize_linear(dx, scale, bias);

	// cpu based calculation (expected result)
	auto hx = dx.copy_to_host();
	auto hy = tensor_quantize_linear(hx, scale, bias);

	// compare
	auto hy_from_cuda = dy.copy_to_host();

	bool eq = elementwise_compatible(hy, hy_from_cuda);  // checks the sizes
	if (eq)
	{
		int8* expected = hy.buffer();
		int8* actual = hy_from_cuda.buffer();

		int length = hy.size();

		for (int ix = 0; ix < length; ++ix)
		{
			eq = eq && (expected[ix] == actual[ix]);

			if (!eq)
			{
				std::cout << (int32)expected[ix] << " " << (int32)actual[ix] << "\n";
			}
		}
	}

	std::cout << "TestCase [test_quant_lin_f32_i8]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_dequant_lin_i8_f32()
{
	// cuda based calculation
	auto dx = crt_random_tensor<int8, CUDA>({ 18, 64 }, 21);
	float32 scale = 0.8f;
	int8 bias = 14;

	auto dy = tensor_dequantize_linear(dx, scale, bias);

	// cpu based calculation (expected result)
	auto hx = dx.copy_to_host();
	auto hy = tensor_dequantize_linear(hx, scale, bias);

	// compare
	auto hy_from_cuda = dy.copy_to_host();

	bool eq = elementwise_compatible(hy, hy_from_cuda);  // checks the sizes
	if (eq)
	{
		float32* expected = hy.buffer();
		float32* actual = hy_from_cuda.buffer();

		int length = hy.size();

		for (int ix = 0; ix < length; ++ix)
		{
			eq = eq && std::abs(expected[ix] - actual[ix]) < 0.001f;

			if (!eq)
			{
				std::cout << expected[ix] << " " << actual[ix] << "\n";
			}
		}
	}

	std::cout << "TestCase [test_dequant_lin_i8_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_qmm_i8_f32()
{
	// cuda based calculation
	auto dta = crt_random_tensor<int8, CUDA>({ 40, 30 }, 12);
	auto dtb = crt_random_tensor<int8, CUDA>({ 30, 50 }, 29);

	float32 sa = 1.8f;
	int8 zpa = 56;
	float32 sb = 3.4f;
	int8 zpb = 87;
	float32 sy = 1.4f;
	int8 zpy = 4;

	auto dtc = tensor_qmm(
		dta, dtb,
		sa, zpa,
		sb, zpb,
		sy, zpy);

	// cpu based calculation (expected result)
	auto hta = dta.copy_to_host();
	auto htb = dtb.copy_to_host();
	auto htc = tensor_qmm(
		hta, htb,
		sa, zpa,
		sb, zpb,
		sy, zpy);

	// compare
	auto htc_from_cuda = dtc.copy_to_host();

	bool eq = elementwise_compatible(htc, htc_from_cuda);  // checks the sizes
	if (eq)
	{
		int8* expected = htc.buffer();
		int8* actual = htc_from_cuda.buffer();

		int length = htc.size();

		for (int ix = 0; ix < length; ++ix)
		{
			eq = eq && (expected[ix] == actual[ix]);

			if (!eq)
			{
				std::cout << expected[ix] << " " << actual[ix] << "\n";
			}
		}
	}

	std::cout << "TestCase [test_qmm_i8_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}
