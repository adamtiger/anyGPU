#include "ext_torch_tests.hpp"
#include "test_tools.hpp"
#include "safetensors_file.hpp"
#include "tensor.hpp"
#include "attention.hpp"
#include "dat_file.hpp"
#include <filesystem>

const std::filesystem::path artifact_folder_path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts";

void external_test_sdp_fwd_f32()
{
	auto path = artifact_folder_path / "test_sdp_fwd_nomask_noscore_f32_16_64";

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
	auto path = artifact_folder_path / "test_sdp_bwd_nomask_noscore_f32_16_64";

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
	auto path = artifact_folder_path / "test_softmax_bwd_f32_16_64";

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


void external_test_sf_data_reading()
{
	// comparator
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	bool eq = true;

	// gather test file names
	std::unordered_map<std::string, Tensor<float32, CPU>> expected_tensors;
	const std::filesystem::path sf_exp_tensors_folder{ artifact_folder_path / "test_sf_diffuser_tensors" };
	for (auto const& sf_tensor_path : std::filesystem::directory_iterator{ sf_exp_tensors_folder })
	{
		const std::string sf_tensor_name = sf_tensor_path.path().stem().string();
		expected_tensors[sf_tensor_name] = load_tensor(sf_tensor_path.path().string());
	}

	// read tensors from files
	std::string path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts\\safetensors\\diffusion_pytorch_model.safetensors";
	std::vector<Tensor<float32, CPU>> tensors;
	sft_read_tensors(path, tensors);

	for (auto& t : tensors)
	{
		if (expected_tensors.contains(t.name))
		{
			auto& expected = expected_tensors.at(t.name);
			eq = eq && cmp(expected, t);
		}
	}

	std::cout << "TestCase [external_test_sf_data_reading]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_layer_norm_fwd_f32()
{
	auto path = artifact_folder_path / "test_layer_norm_fwd_f32";

	// read tensors from files
	auto hx = load_tensor((path / "x.dat").string());
	auto hw = load_tensor((path / "w.dat").string());
	auto hb = load_tensor((path / "b.dat").string());
	auto exp_hy = load_tensor((path / "y.dat").string());

	auto act_hy_cpu = tensor_layer_norm(hx, 2, hw, hb, 2e-3f);


	auto dx = hx.copy_to_cuda();
	auto dw = hw.copy_to_cuda();
	auto db = hb.copy_to_cuda();

	auto act_dy_cuda = tensor_layer_norm(dx, 2, dw, db, 2e-3f);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cpu
	bool eq = cmp(exp_hy, act_hy_cpu);
	std::cout << "TestCase [external_test_layer_norm_fwd_f32 - CPU]: " << (eq ? "PASSED" : "FAILED") << "\n";

	// test cuda
	eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_layer_norm_fwd_f32 - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_rms_norm_fwd_f32()
{
	auto path = artifact_folder_path / "test_rms_norm_fwd_f32";

	// read tensors from files
	auto hx = load_tensor((path / "x.dat").string());
	auto hw = load_tensor((path / "w.dat").string());
	auto exp_hy = load_tensor((path / "y.dat").string());

	auto act_hy_cpu = tensor_rms_norm(hx, 2, hw, 2e-3f);


	auto dx = hx.copy_to_cuda();
	auto dw = hw.copy_to_cuda();

	auto act_dy_cuda = tensor_rms_norm(dx, 2, dw, 2e-3f);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cpu
	bool eq = cmp(exp_hy, act_hy_cpu);
	std::cout << "TestCase [external_test_rms_norm_fwd_f32 - CPU]: " << (eq ? "PASSED" : "FAILED") << "\n";

	// test cuda
	eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_rms_norm_fwd_f32 - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_silu_fwd_f32()
{
	auto path = artifact_folder_path / "test_silu_fwd_f32";

	// read tensors from files
	auto hx = load_tensor((path / "x.dat").string());
	auto exp_hy = load_tensor((path / "y.dat").string());
	
	auto act_hy_cpu = tensor_silu(hx);  // cpu test

	auto dx = hx.copy_to_cuda();
	auto act_dy_cuda = tensor_silu(dx);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cpu
	bool eq = cmp(exp_hy, act_hy_cpu);
	std::cout << "TestCase [external_test_silu_fwd_f32 - CPU]: " << (eq ? "PASSED" : "FAILED") << "\n";

	// test cuda
	eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_silu_fwd_f32 - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_embedding_fwd_f32()
{
	auto path = artifact_folder_path / "test_embedding_fwd_f32";

	// read tensors from files
	auto hx = load_tensor<int32>((path / "indices.dat").string());
	auto hw = load_tensor((path / "embedding.dat").string());
	auto exp_hy = load_tensor((path / "y.dat").string());

	auto act_hy_cpu = tensor_embedding(hx, hw);  // cpu test

	auto dx = hx.copy_to_cuda();
	auto dw = hw.copy_to_cuda();
	auto act_dy_cuda = tensor_embedding(dx, dw);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cpu
	bool eq = cmp(exp_hy, act_hy_cpu);
	std::cout << "TestCase [external_test_embedding_fwd_f32 - CPU]: " << (eq ? "PASSED" : "FAILED") << "\n";

	// test cuda
	eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_embedding_fwd_f32 - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}

void external_test_rotary_embedding_fwd_f32()
{
	auto path = artifact_folder_path / "test_rotary_embedding_fwd_f32";

	// read tensors from files
	auto hx = load_tensor((path / "x.dat").string());
	auto exp_hy = load_tensor((path / "y.dat").string());

	auto hf = tensor_precomp_rotary_embedding<float32, CPU>(1024, 64);
	auto act_hy_cpu = tensor_apply_rotary_embedding(hx, hf);  // cpu test

	/*auto dx = hx.copy_to_cuda();
	auto dw = hw.copy_to_cuda();
	auto act_dy_cuda = tensor_embedding(dx, dw);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();*/

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cpu
	bool eq = cmp(exp_hy, act_hy_cpu);
	std::cout << "TestCase [external_test_rotary_embedding_fwd_f32 - CPU]: " << (eq ? "PASSED" : "FAILED") << "\n";

	// test cuda
	/*eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_embedding_fwd_f32 - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";*/
}


void external_test_alt_rotary_embedding_fwd_f32()
{
	auto path = artifact_folder_path / "test_alt_rotary_embedding_fwd_f32";

	// read tensors from files
	auto hx = load_tensor((path / "q.dat").string());
	auto exp_hy = load_tensor((path / "y.dat").string());

	auto hf = tensor_precomp_rotary_embedding<float32, CPU>(1024, 64);
	auto act_hy_cpu = tensor_apply_alt_rotary_embedding(hx, hf);  // cpu test

	/*auto dx = hx.copy_to_cuda();
	auto dw = hw.copy_to_cuda();
	auto act_dy_cuda = tensor_embedding(dx, dw);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();*/

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cpu
	bool eq = cmp(exp_hy, act_hy_cpu);
	std::cout << "TestCase [external_test_alt_rotary_embedding_fwd_f32 - CPU]: " << (eq ? "PASSED" : "FAILED") << "\n";

	// test cuda
	/*eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_alt_rotary_embedding_fwd_f32 - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";*/
}
