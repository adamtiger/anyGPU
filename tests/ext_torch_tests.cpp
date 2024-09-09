#include "ext_torch_tests.hpp"
#include "test_tools.hpp"
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
	auto dy = single_head_attention_fwd(dq, dk, dv);

	// compare
	auto hy_from_cuda = dy.copy_to_host();

	bool eq = elementwise_compatible(hy_from_cuda, hy);  // checks the sizes
	eq = eq && compare_data_buffers(hy_from_cuda, hy);

	std::cout << "TestCase [external_test_sdp_fwd_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_sdp_bwd_f32()
{
	auto path = artifact_folder_path / "sdp_bwd_nomask_noscore_f32_16_64";

	// read tensors from files
	auto hq = load_tensor((path / "q.dat").string());
	auto hk = load_tensor((path / "k.dat").string());
	auto hv = load_tensor((path / "v.dat").string());

	auto h_grad_y = load_tensor((path / "grad_y.dat").string());
	auto h_grad_q = load_tensor((path / "grad_q.dat").string());
	auto h_grad_k = load_tensor((path / "grad_k.dat").string());
	auto h_grad_v = load_tensor((path / "grad_v.dat").string());

	// cuda based calculation
	auto d_grad_y = h_grad_y.copy_to_cuda();
	auto dq = hq.copy_to_cuda();
	auto dk = hk.copy_to_cuda();
	auto dv = hv.copy_to_cuda();

	SDPGradient grads = single_head_attention_bwd(dq, dk, dv, d_grad_y);

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
	{
		bool eq = elementwise_compatible(actual, expected);  // checks the sizes
		eq = eq && compare_data_buffers(actual, expected);
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


void external_test_cpu_softmax_bwd_f32()
{
	auto path = artifact_folder_path / "softmax_bwd_f32_16_64";

	// read tensors from files
	auto hx = load_tensor((path / "x.dat").string());

	auto h_grad_y = load_tensor((path / "grad_y.dat").string());
	auto h_grad_x = load_tensor((path / "grad_x.dat").string());

	auto hy = tensor_softmax(hx);
	auto grad_x = tensor_softmax_bwd(hy, h_grad_y);

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
	{
		bool eq = elementwise_compatible(expected, actual);  // checks the sizes
		eq = eq && compare_data_buffers(actual, expected);
		return eq;
	};

	bool eq = cmp(h_grad_x, grad_x);

	std::cout << "TestCase [external_test_cpu_softmax_bwd_f32]: " << (eq ? "PASSED" : "FAILED") << "\n";
}