#include "ext_torch_tests.hpp"
#include "tensor.hpp"
#include "attention.hpp"
#include <filesystem>

const std::filesystem::path artifact_folder_path = "C:\\Data\\AI\\projects\\anyGPU\\artifacts";

void external_test_sdp_fwd_f32()
{
	auto path = artifact_folder_path / "sdp_nomask_noscore_f32_16_64";

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

