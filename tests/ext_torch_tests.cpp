#include "ext_torch_tests.hpp"
#include "tensor.hpp"
#include "attention.hpp"
#include <filesystem>

const std::filesystem::path artifact_folder_path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts";

void external_test_sdp_fwd_f32()
{
	auto path = artifact_folder_path / "sdp_fwd_nomask_noscore_f32_16_64";

	// read tensors from files
	auto hq = load_tensor((path / "q.dat").string());
	auto hk = load_tensor((path / "k.dat").string());
	auto hv = load_tensor((path / "v.dat").string());
	auto hy = load_tensor((path / "y.dat").string());

	// cuda based calculation
	auto dq = hq.copy_to_cuda();
	auto dk = hk.copy_to_cuda();
	auto dv = hv.copy_to_cuda();
	auto dy = single_head_attention_fwd<float32, CUDA, NONE, FULL>(dq, dk, dv);

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

	std::cout << "TestCase [external_test_sdp_fwd_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_sdp_bwd_f32()
{
	auto path = artifact_folder_path / "sdp_bwd_nomask_noscore_f32_16_64";

	// read tensors from files
	auto hq = load_tensor((path / "q.dat").string());
	auto hk = load_tensor((path / "k.dat").string());
	auto hv = load_tensor((path / "v.dat").string());

	auto h_grad_q = load_tensor((path / "grad_q.dat").string());
	auto h_grad_k = load_tensor((path / "grad_k.dat").string());
	auto h_grad_v = load_tensor((path / "grad_v.dat").string());

	// cuda based calculation
	auto dq = hq.copy_to_cuda();
	auto dk = hk.copy_to_cuda();
	auto dv = hv.copy_to_cuda();

	SDPGradient grads = single_head_attention_bwd<float32, CUDA, NONE, FULL>(dq, dk, dv);

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

	auto h_grad_q_from_cuda = grads.grad_q.copy_to_host();
	eq = eq && cmp(h_grad_q, h_grad_q_from_cuda);

	auto h_grad_k_from_cuda = grads.grad_k.copy_to_host();
	eq = eq && cmp(h_grad_k, h_grad_k_from_cuda);

	auto h_grad_v_from_cuda = grads.grad_v.copy_to_host();
	eq = eq && cmp(h_grad_v, h_grad_v_from_cuda);

	std::cout << "TestCase [external_test_sdp_bwd_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}
