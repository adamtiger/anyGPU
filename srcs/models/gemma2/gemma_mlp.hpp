#ifndef __GEMMA_MLP__
#define __GEMMA_MLP__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "ops.hpp"


template<FloatingPointType dtype, Device device>
struct GemmaMLPweights
{
	Tensor<dtype, device> gate_proj_weight;
	Tensor<dtype, device> up_proj_weight;
	Tensor<dtype, device> down_proj_weight;

	void load_weights(
		const std::string& path_gate_proj_weight,
		const std::string& path_up_proj_weight,
		const std::string& path_down_proj_weight)
	{
		auto load_w = [&](const std::string& path_w)
		{
			return copy_to_device<dtype, CPU, device>(tensor_transp(load_tensor<dtype>(path_w)));
		};

		gate_proj_weight = load_w(path_gate_proj_weight);
		up_proj_weight = load_w(path_up_proj_weight);
		down_proj_weight = load_w(path_down_proj_weight);
	}
};

template<FloatingPointType dtype, Device device>
inline Tensor<dtype, device> tensor_gemma_mlp(
	const GemmaMLPweights<dtype, device>& mlp_weights,
	const Tensor<dtype, device>& x)
{
	auto gated_x = tensor_linear(x, mlp_weights.gate_proj_weight);
	auto act_gated_x = tensor_gelu(gated_x, true);

	auto up_x = tensor_linear(x, mlp_weights.up_proj_weight);

	auto comb_x = tensor_mul(act_gated_x, up_x);

	auto y = tensor_linear(comb_x, mlp_weights.down_proj_weight);

	return y;
}

#endif  // __GEMMA_MLP__
