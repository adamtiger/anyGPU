#include "ext_zamba2_tests.hpp"
#include "test_tools.hpp"
#include "dat_file.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include <filesystem>

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

