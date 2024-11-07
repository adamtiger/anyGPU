#ifndef __ZAMBA_MLP__
#define __ZAMBA_MLP__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "transp_ops.hpp"
#include "binary_ops.hpp"
#include "mm_ops.hpp"
#include "zamba_glu.hpp"

template<FloatingPointType dtype, Device device>
struct ZambaMLPweights
{
	Tensor<dtype, device> fc1_weight;
	Tensor<dtype, device> fc2_weight;
	std::vector<Tensor<dtype, device>> fc1_lora_A_weights;
	std::vector<Tensor<dtype, device>> fc1_lora_B_weights;

	void load_weights(
		const std::string& path_fc1_weight,
		const std::string& path_fc2_weight,
		const std::vector<std::string>& paths_fc1_lora_A_weights,
		const std::vector<std::string>& paths_fc1_lora_B_weights)
	{
		auto load_w = [&](const std::string& path_w)
		{
			return tensor_transp(load_tensor<dtype>(path_w).copy_to_device<device>());
		};

		fc1_weight = load_w(path_fc1_weight);
		fc2_weight = load_w(path_fc2_weight);

		ACASSERT(
			paths_fc1_lora_A_weights.size() == paths_fc1_lora_B_weights.size(), 
			"loraA and loraB needs to have the same number of blocks"
		);

		size_t nw = paths_fc1_lora_A_weights.size();
		fc1_lora_A_weights.reserve(nw);
		fc1_lora_B_weights.reserve(nw);
		for (size_t ix = 0; ix < nw; ++ix)
		{
			fc1_lora_A_weights.push_back(load_w(paths_fc1_lora_A_weights[ix]));

			fc1_lora_B_weights.push_back(load_w(paths_fc1_lora_B_weights[ix]));
		}
	}
};

template<FloatingPointType dtype, Device device>
inline Tensor<dtype, device> tensor_zamba_mlp(
	const ZambaMLPweights<dtype, device>& mlp_weights,
	const Tensor<dtype, device>& hidden_states,
	const int fwd_layer_idx)
{
	auto lora_A_output = tensor_linear(hidden_states, mlp_weights.fc1_lora_A_weights[fwd_layer_idx]);
	auto lora_output = tensor_linear(lora_A_output, mlp_weights.fc1_lora_B_weights[fwd_layer_idx]);

	auto fc1_output = tensor_linear(hidden_states, mlp_weights.fc1_weight);

	auto intermediate_parallel = tensor_add(fc1_output, lora_output);

	auto glu_output = tensor_zamba_glu(intermediate_parallel);

	auto output = tensor_linear(glu_output, mlp_weights.fc2_weight);

	return output;
}

#endif  // __ZAMBA_MLP__
