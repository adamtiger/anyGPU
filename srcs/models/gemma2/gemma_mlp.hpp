#ifndef __GEMMA_MLP__
#define __GEMMA_MLP__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "ops.hpp"

#include "mlp_gemma2_fused_upproj.cuh"
#include "mlp_gemma2_dp_linear.cuh"


template<FloatingPointType dtype, Device device>
struct GemmaMLPweights
{
	Tensor<dtype, device> gate_proj_weight;
	Tensor<dtype, device> up_proj_weight;
	Tensor<dtype, device> down_proj_weight;

	void load_weights(
		const std::string& path_gate_proj_weight,
		const std::string& path_up_proj_weight,
		const std::string& path_down_proj_weight,
		const bool transpose=true)
	{
		auto load_w = [&](const std::string& path_w, const bool transpose)
		{
			auto t = load_tensor<dtype>(path_w);
			if (transpose)
			{
				t = tensor_transp(t);
			}
			return copy_to_device<dtype, CPU, device>(t);
		};

		gate_proj_weight = load_w(path_gate_proj_weight, transpose);
		up_proj_weight = load_w(path_up_proj_weight, transpose);
		down_proj_weight = load_w(path_down_proj_weight, transpose);
	}

	void set_weights(
		Tensor<dtype, device>& mlp_gate_proj_weight,
		Tensor<dtype, device>& mlp_up_proj_weight,
		Tensor<dtype, device>& mlp_down_proj_weight)
	{
		gate_proj_weight = tensor_transp(mlp_gate_proj_weight);
		up_proj_weight = tensor_transp(mlp_up_proj_weight);
		down_proj_weight = tensor_transp(mlp_down_proj_weight);
	}
};

template<FloatingPointType dtype, Device device, int variant>
inline Tensor<dtype, device> tensor_gemma_mlp_fused_uprpoj(
	const GemmaMLPweights<dtype, device>& mlp_weights,
	const Tensor<dtype, device>& x)
{
	Tensor<dtype, device> y;

	if constexpr (device == CUDA && variant == 1)
	{
		Shape y_shape = x.shape;
		y_shape[x.dim - 1] = x.shape[x.dim - 1] * 4;
		Tensor<dtype, device> out(x.dim, y_shape);

		cu_mlp_gemma2_fused_upproj_f32_v1(
			x, mlp_weights.gate_proj_weight, mlp_weights.up_proj_weight, out
		);

		y = out;
	}
	else if constexpr (device == CUDA && variant == 2)
	{
		Shape y_shape = x.shape;
		y_shape[x.dim - 1] = x.shape[x.dim - 1] * 4;
		Tensor<dtype, device> out(x.dim, y_shape);

		cu_mlp_gemma2_fused_upproj_f32_v2(
			x, mlp_weights.gate_proj_weight, mlp_weights.up_proj_weight, out
		);

		y = out;
	}
	else  // default (for any device)
	{
		auto gated_x = tensor_linear(x, mlp_weights.gate_proj_weight);
		auto act_gated_x = tensor_gelu(gated_x, true);

		auto up_x = tensor_linear(x, mlp_weights.up_proj_weight);

		y = tensor_mul(act_gated_x, up_x);
	}

	return y;
}


template<FloatingPointType dtype, Device device, int variant>
inline Tensor<dtype, device> tensor_gemma_mlp_dp_linear(
	const GemmaMLPweights<dtype, device>& mlp_weights,
	const Tensor<dtype, device>& x)
{
	Tensor<dtype, device> y;

	if constexpr (device == CUDA && variant == 1)
	{
		Shape y_shape = x.shape;
		y_shape[x.dim - 1] = x.shape[x.dim - 1] / 4;  // 9216 -> 2304
		Tensor<dtype, device> out(x.dim, y_shape);

		cu_mlp_gemma2_dp_linear_f32_v1(
			x, mlp_weights.down_proj_weight, out
		);

		y = out;
	}
	else if constexpr (device == CUDA && variant == 3)
	{
		Shape y_shape = x.shape;
		y_shape[x.dim - 1] = x.shape[x.dim - 1] / 4;  // 9216 -> 2304
		Tensor<dtype, device> out(x.dim, y_shape);

		cu_mlp_gemma2_dp_linear_f32_v3(
			x, mlp_weights.down_proj_weight, out
		);

		y = out;
	}
	else  // default (for any device)
	{
		y = tensor_linear(x, mlp_weights.down_proj_weight);
	}

	return y;
}


template<FloatingPointType dtype, Device device>
inline Tensor<dtype, device> tensor_gemma_mlp(
	const GemmaMLPweights<dtype, device>& mlp_weights,
	const Tensor<dtype, device>& x)
{
	auto comb_x = tensor_gemma_mlp_fused_uprpoj<dtype, device, 2>(mlp_weights, x);

	auto y = tensor_gemma_mlp_dp_linear<dtype, device, 1>(mlp_weights, comb_x);

	return y;
}

#endif  // __GEMMA_MLP__
