#include "gemma2_tests.hpp"
#include "test_tools.hpp"
#include "dat_file.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include <filesystem>

#include "gemma_mlp.hpp"


const std::filesystem::path artifact_folder_path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts\\xgemma2_tests";


void external_test_gemma2_decoder_mlp()
{
	auto path = artifact_folder_path / "test_gemma2decoder_mlp";

	// read tensors from files
	auto hx = load_tensor((path / "in_0.dat").string());
	auto exp_hy = load_tensor((path / "out_0.dat").string());

	GemmaMLPweights<float32, CUDA> mlp_weights;
	mlp_weights.load_weights(
		(path / "gemma2decoder_mlp.gemma2mlp.gate_proj.weight.dat").string(),
		(path / "gemma2decoder_mlp.gemma2mlp.up_proj.weight.dat").string(),
		(path / "gemma2decoder_mlp.gemma2mlp.down_proj.weight.dat").string()
	);

	auto dx = hx.copy_to_cuda();
	auto act_dy_cuda = tensor_gemma_mlp(mlp_weights, dx);
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
	std::cout << "TestCase [external_test_gemma2_decoder_mlp - CUDA]: " << (eq ? "PASSED" : "FAILED") << "\n";
}
