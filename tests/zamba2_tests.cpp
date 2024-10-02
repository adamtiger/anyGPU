#include "zamba2_tests.hpp"
#include "test_tools.hpp"
#include "dat_file.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include <filesystem>

#include "zamba_glu.hpp"
#include "zamba_mlp.hpp"


const std::filesystem::path artifact_folder_path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts\\zamba2_tests";

void external_test_zamba2_model_rmsnorm()
{
	auto path = artifact_folder_path / "test_zamba2model_rmsnorm";

	// read tensors from files
	auto hw = load_tensor((path / "zamba2model_rmsnorm.zamba2rmsnorm.weight.dat").string());
	auto hx = load_tensor((path / "in_0.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	auto dx = hx.copy_to_cuda();
	auto dw = hw.copy_to_cuda();
	auto act_dy_cuda = tensor_rms_norm<float32>(dx, -1, dw, 1e-5f);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cuda
	bool eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_zamba2_model_rmsnorm - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void external_test_zamba2_attndeco_mlp()
{
	auto path = artifact_folder_path / "test_zamba2AttenDecodLy_mlp";

	// read tensors from files
	auto hx = load_tensor((path / "in_0.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	ZambaMLPweights<float32, CUDA> mlp_weights;
	mlp_weights.load_weights(
		(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1.weight.dat").string(),
		(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc2.weight.dat").string(),
		{
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.0.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.1.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.2.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.3.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.4.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_A_list.5.weight.dat").string()
		},
		{
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.0.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.1.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.2.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.3.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.4.weight.dat").string(),
			(path / "zamba2AttenDecodLy_mlp.zamba2mlp.linear_fc1_lora_B_list.5.weight.dat").string()
		}
	);

	const int fwd_layer_idx = 0;

	auto dx = hx.copy_to_cuda();
	auto act_dy_cuda = tensor_zamba_mlp(mlp_weights, dx, fwd_layer_idx);
	auto act_hy_cuda = act_dy_cuda.copy_to_host();

	// compare
	auto cmp = [&](const Tensor<float32, CPU>& expected, const Tensor<float32, CPU>& actual)
		{
			bool eq = elementwise_compatible(expected, actual);  // checks the sizes
			eq = eq && compare_data_buffers(actual, expected);
			return eq;
		};

	// test cuda
	bool eq = cmp(exp_hy, act_hy_cuda);
	std::cout << "TestCase [external_test_zamba2_attndeco_mlp - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}


void test_zamba2_glu()
{
	// cuda based calculation
	auto dta = crt_random_tensor<float32, CUDA>({ 2, 4, 5, 2048 }, 11);
	auto dtc = tensor_zamba_glu(dta);

	// cpu based calculation (expected result)
	auto hta = dta.copy_to_host();
	auto htc = tensor_zamba_glu(hta);

	// compare
	auto htc_from_cuda = dtc.copy_to_host();

	bool eq = elementwise_compatible(htc_from_cuda, htc);  // checks the sizes
	eq = eq && compare_data_buffers(htc_from_cuda, htc);

	std::cout << "TestCase [test_zamba2_glu]: " << (eq ? "PASSED" : "FAILED") << "\n";
}
