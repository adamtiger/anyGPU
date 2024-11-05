#ifndef __GEMMA_MODEL__
#define __GEMMA_MODEL__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "dat_file.hpp"
#include "ops.hpp"

#include "gemma_update_mask.hpp"
#include "gemma_decoder.hpp"
#include "gemma_config.hpp"


template<FloatingPointType dtype>
struct GemmaModelweights
{
	std::vector<GemmaDecoderweights<dtype>> decoder_layers;
	Tensor<dtype, CUDA> embedding_data;
	Tensor<dtype, CUDA> pos_rmsnorm_weight;

	void load_weights(const std::string& weights_folder_path)
	{
		// constants
		const int num_decoder_weights = 11;
		const int offset_layer_id_str = 8;  // length of .layers.

        // paths
		std::string path_embedding_data;
		std::string path_pos_rmsnorm_weight;
		std::vector<std::string> paths_decoder_layer_weight;  // fixed order, 2d table as 1d array

		// gather the individual weight paths
		const std::filesystem::path weights_folder(weights_folder_path);

		//   regexes for finding the weights
		const std::regex embedding_data_regex("\\.embed_tokens\\.weight\\.dat$");
		const std::regex pos_rmsnorm_weight_regex("\\.norm\\.weight\\.dat$");

		const std::regex gd_in_rmsnorm_weight_regex("\\.layers\\.\\d+\\.input_layernorm\\.weight\\.dat$");
		const std::regex gd_preff_rmsnorm_weight_regex("\\.layers\\.\\d+\\.pre_feedforward_layernorm\\.weight\\.dat$");
		const std::regex gd_postf_rmsnorm_weight_regex("\\.layers\\.\\d+\\.post_feedforward_layernorm\\.weight\\.dat$");
		const std::regex gd_post_attn_rmsnorm_weight_regex("\\.layers\\.\\d+\\.post_attention_layernorm\\.weight\\.dat$");

		const std::regex sdpa_q_proj_weight_regex("\\.layers\\.\\d+\\.self_attn\\.q_proj\\.weight\\.dat$");
		const std::regex sdpa_k_proj_weight_regex("\\.layers\\.\\d+\\.self_attn\\.k_proj\\.weight\\.dat$");
		const std::regex sdpa_v_proj_weight_regex("\\.layers\\.\\d+\\.self_attn\\.v_proj\\.weight\\.dat$");
		const std::regex sdpa_o_proj_weight_regex("\\.layers\\.\\d+\\.self_attn\\.o_proj\\.weight\\.dat$");

		const std::regex mlp_gate_proj_weight_regex("\\.layers\\.\\d+\\.mlp\\.gate_proj\\.weight\\.dat$");
		const std::regex mlp_up_proj_weight_regex("\\.layers\\.\\d+\\.mlp\\.up_proj\\.weight\\.dat$");
		const std::regex mlp_down_proj_weight_regex("\\.layers\\.\\d+\\.mlp\\.down_proj\\.weight\\.dat$");

		const std::regex layer_weight_regex("\\.layers\\.\\d+");

		std::vector<std::regex> decoder_weight_regexes =
		{
			gd_in_rmsnorm_weight_regex,
			gd_preff_rmsnorm_weight_regex,
			gd_postf_rmsnorm_weight_regex,
			gd_post_attn_rmsnorm_weight_regex,
			sdpa_q_proj_weight_regex,
			sdpa_k_proj_weight_regex,
			sdpa_v_proj_weight_regex,
			sdpa_o_proj_weight_regex,
			mlp_gate_proj_weight_regex,
			mlp_up_proj_weight_regex,
			mlp_down_proj_weight_regex
		};

		// calculate the number of layers from the number of files
		int num_layer_weight_files = 0;
		for (auto const& entry : std::filesystem::directory_iterator{ weights_folder })
		{
			if (entry.is_regular_file())
			{
				auto file_name = entry.path().filename().string();
				if (std::regex_search(file_name, layer_weight_regex))
				{
					num_layer_weight_files++;
				}
			}
		}

		const int num_layers = num_layer_weight_files / num_decoder_weights;

		paths_decoder_layer_weight.resize(num_layer_weight_files);


		// identify the passes and save them
		for (auto const& entry : std::filesystem::directory_iterator{ weights_folder })
		{
			if (entry.is_regular_file())
			{
				auto weight_path = entry.path().string();
				auto file_name = entry.path().filename().string();

				// examine the file name and save it
				if (std::regex_search(file_name, embedding_data_regex))
				{
					path_embedding_data = weight_path;
				}
				else if (std::regex_search(file_name, pos_rmsnorm_weight_regex))
				{
					path_pos_rmsnorm_weight = weight_path;
				}
				else
				{
					std::smatch sm;

					for (int wix = 0; wix < num_decoder_weights; ++wix)
					{
						if (std::regex_search(file_name, sm, decoder_weight_regexes[wix]))
						{
							auto matched_str = sm[0].str();

							// get the layer id
							std::regex_search(matched_str, sm, layer_weight_regex);
							auto layer_num_str = sm[0].str().substr(offset_layer_id_str);
							int layer_id = std::atoi(layer_num_str.c_str());

							// insert new element
							paths_decoder_layer_weight[layer_id * num_decoder_weights + wix] = weight_path;
						}
					}
				}
			}
		}

		// load the weights
		auto load_w = [&](const std::string& path_w)
		{
			return load_tensor<dtype>(path_w).copy_to_device<CUDA>();
		};

		embedding_data = load_w(path_embedding_data);
		pos_rmsnorm_weight = load_w(path_pos_rmsnorm_weight);

		decoder_layers.resize(num_layers);

		for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx)
		{
			auto& decoder_weights = decoder_layers[layer_idx];

			int offset = layer_idx * num_decoder_weights;

			decoder_weights.load_weights(
				paths_decoder_layer_weight[offset + 0],
				paths_decoder_layer_weight[offset + 1],
				paths_decoder_layer_weight[offset + 2],
				paths_decoder_layer_weight[offset + 3],
				paths_decoder_layer_weight[offset + 4],
				paths_decoder_layer_weight[offset + 5],
				paths_decoder_layer_weight[offset + 6],
				paths_decoder_layer_weight[offset + 7],
				paths_decoder_layer_weight[offset + 8],
				paths_decoder_layer_weight[offset + 9],
				paths_decoder_layer_weight[offset + 10]
			);
		}
	}
};

template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_gemma_model(  // Gemma2Model
	const GemmaConfig& config,
	const GemmaModelweights<dtype>& model_weights,
	GemmaKVcache& kv_cache,
	const Tensor<int32, CUDA>& input_ids,
	const Tensor<dtype, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& position_ids)
{
	// embed the input ids (calculated from tokenizer)
	auto inp_embeds = tensor_embedding(input_ids, model_weights.embedding_data);

	// TODO: update_causal mask (requires implementation!)
	/*auto causal_mas = tensor_gemma_update_mask(
		attention_mask,
		inp_embeds,
		cache_position,  ???
		config.target_length
	);*/


	// normalize hidden states
	dtype norm_factor = static_cast<dtype>(sqrtf(static_cast<float32>(config.hidden_size)));
	auto hidden_states = tensor_mul(inp_embeds, norm_factor);

	// execute the decoder layers (one after each other)
	for (auto& decoder_weights : model_weights.decoder_layers)
	{
		auto layer_outputs = tensor_gemma_decoder(
			config,
			decoder_weights,
			kv_cache,
			hidden_states,
			attention_mask,
			position_ids);

		hidden_states = layer_outputs;  // TODO: may be this will be a list to return more values! (or return on the argument!)
	}

	// calculate rms norm
	auto y = tensor_rms_norm(hidden_states, -1, model_weights.pos_rmsnorm_weight, config.rms_norm_eps, true);


	// TODO: store required additional outputs (e.g. all hidden states, attentions etc.) (is this required???, what is the purpose? saving computation and memory?)

	return y;
}

#endif // __GEMMA_MODEL__
