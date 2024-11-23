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
			return copy_to_device<dtype, CPU, CUDA>(load_tensor<dtype>(path_w));
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

	void load_weights_from_safetensors(const std::string& weights_folder_path)
	{
		// read and load all tensors from the safetensor files
		std::vector<Tensor<dtype, CUDA>> loaded_tensors;
		_load_tensors_from_safetensors(weights_folder_path, loaded_tensors);

		// identify and assign the tensors

		// constants
		const int num_decoder_weights = 11;
		const int offset_layer_id_str = 8;  // length of .layers.

		// paths
		int index_embedding_data;
		int index_pos_rmsnorm_weight;
		std::vector<int> indices_decoder_layer_weight;  // fixed order, 2d table as 1d array

		//   regexes for finding the weights
		const std::regex embedding_data_regex("\\.embed_tokens\\.weight$");
		const std::regex pos_rmsnorm_weight_regex("\\.norm\\.weight$");

		const std::regex gd_in_rmsnorm_weight_regex("\\.layers\\.\\d+\\.input_layernorm\\.weight$");
		const std::regex gd_preff_rmsnorm_weight_regex("\\.layers\\.\\d+\\.pre_feedforward_layernorm\\.weight$");
		const std::regex gd_postf_rmsnorm_weight_regex("\\.layers\\.\\d+\\.post_feedforward_layernorm\\.weight$");
		const std::regex gd_post_attn_rmsnorm_weight_regex("\\.layers\\.\\d+\\.post_attention_layernorm\\.weight$");

		const std::regex sdpa_q_proj_weight_regex("\\.layers\\.\\d+\\.self_attn\\.q_proj\\.weight$");
		const std::regex sdpa_k_proj_weight_regex("\\.layers\\.\\d+\\.self_attn\\.k_proj\\.weight$");
		const std::regex sdpa_v_proj_weight_regex("\\.layers\\.\\d+\\.self_attn\\.v_proj\\.weight$");
		const std::regex sdpa_o_proj_weight_regex("\\.layers\\.\\d+\\.self_attn\\.o_proj\\.weight$");

		const std::regex mlp_gate_proj_weight_regex("\\.layers\\.\\d+\\.mlp\\.gate_proj\\.weight$");
		const std::regex mlp_up_proj_weight_regex("\\.layers\\.\\d+\\.mlp\\.up_proj\\.weight$");
		const std::regex mlp_down_proj_weight_regex("\\.layers\\.\\d+\\.mlp\\.down_proj\\.weight$");

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
		int num_tensors = static_cast<int>(loaded_tensors.size());
		int num_layer_weight_files = num_tensors - 2;
		const int num_layers = num_layer_weight_files / num_decoder_weights;

		indices_decoder_layer_weight.resize(num_layer_weight_files);

		// identify the tensors and save them
		for (int tix = 0; tix < num_tensors; ++tix)
		{
			Tensor<dtype, CUDA>& tensor = loaded_tensors[tix];
			auto& tensor_name = tensor.name;

			// examine the file name and save it
			if (std::regex_search(tensor_name, embedding_data_regex))
			{
				index_embedding_data = tix;
			}
			else if (std::regex_search(tensor_name, pos_rmsnorm_weight_regex))
			{
				index_pos_rmsnorm_weight = tix;
			}
			else
			{
				std::smatch sm;

				for (int wix = 0; wix < num_decoder_weights; ++wix)
				{
					if (std::regex_search(tensor_name, sm, decoder_weight_regexes[wix]))
					{
						auto matched_str = sm[0].str();

						// get the layer id
						std::regex_search(matched_str, sm, layer_weight_regex);
						auto layer_num_str = sm[0].str().substr(offset_layer_id_str);
						int layer_id = std::atoi(layer_num_str.c_str());

						// insert new element
						indices_decoder_layer_weight[layer_id * num_decoder_weights + wix] = tix;
					}
				}
			}
		}

		// set the weights
		embedding_data = loaded_tensors[index_embedding_data];
		pos_rmsnorm_weight = loaded_tensors[index_pos_rmsnorm_weight];

		decoder_layers.resize(num_layers);

		for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx)
		{
			auto& decoder_weights = decoder_layers[layer_idx];

			int offset = layer_idx * num_decoder_weights;

			decoder_weights.set_weights(
				loaded_tensors[indices_decoder_layer_weight[offset + 0]],
				loaded_tensors[indices_decoder_layer_weight[offset + 1]],
				loaded_tensors[indices_decoder_layer_weight[offset + 2]],
				loaded_tensors[indices_decoder_layer_weight[offset + 3]],
				loaded_tensors[indices_decoder_layer_weight[offset + 4]],
				loaded_tensors[indices_decoder_layer_weight[offset + 5]],
				loaded_tensors[indices_decoder_layer_weight[offset + 6]],
				loaded_tensors[indices_decoder_layer_weight[offset + 7]],
				loaded_tensors[indices_decoder_layer_weight[offset + 8]],
				loaded_tensors[indices_decoder_layer_weight[offset + 9]],
				loaded_tensors[indices_decoder_layer_weight[offset + 10]]
			);
		}
	}

private:

	void _load_tensors_from_safetensors(
		const std::string& sf_weights_folder_path,
		std::vector<Tensor<dtype, CUDA>>& loaded_tensors)
	{
		// find the safetensor file paths
		std::vector<std::string> sf_paths;
		{
			const std::regex sf_extension("\\.safetensors$");

			for (auto const& entry : std::filesystem::directory_iterator{ sf_weights_folder_path })
			{
				if (entry.is_regular_file())
				{
					auto file_name = entry.path().filename().string();
					if (std::regex_search(file_name, sf_extension))
					{
						auto sfp = entry.path().string();
						sf_paths.push_back(sfp);
					}
				}
			}
		}
		
		// read the tensors from the safetensor files
		for (const auto& sf_path : sf_paths)
		{
			std::vector<Tensor<dtype, CUDA>> tensors_partial;
			sft_read_tensors(sf_path, tensors_partial);

			// append to the loaded_tensors
			size_t increased_size = loaded_tensors.size() + tensors_partial.size();
			loaded_tensors.resize(increased_size);
			
			loaded_tensors.insert(
				loaded_tensors.begin() + loaded_tensors.size(),
				tensors_partial.begin(), 
				tensors_partial.end()
			);
		}		
	}

	void _load_tensors_from_datfiles(
		const std::string& weights_folder_path,
		std::vector<Tensor<dtype, CUDA>>& loaded_tensors)
	{

	}
};

template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> tensor_gemma_model(  // Gemma2Model
	const GemmaConfig& config,
	const GemmaModelweights<dtype>& model_weights,
	GemmaKVcache& kv_cache,
	const Tensor<int32, CUDA>& input_ids,
	const Tensor<dtype, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& position_ids,
	const Tensor<int32, CUDA>& cache_position)
{
	// embed the input ids (calculated from tokenizer)
	auto inp_embeds = tensor_embedding(input_ids, model_weights.embedding_data);

	// update_causal mask
	auto& causal_mask = attention_mask; 

	// normalize hidden states
	dtype norm_factor = static_cast<dtype>(sqrtf(static_cast<float32>(config.hidden_size)));
	auto hidden_states = tensor_mul(inp_embeds, norm_factor);

	// execute the decoder layers (one after each other)
	int32 layer_idx = 0;
	for (auto& decoder_weights : model_weights.decoder_layers)
	{
		int32 sliding_window = (layer_idx % 2 == 0 ? config.sliding_window : -1);

		auto layer_outputs = tensor_gemma_decoder(
			config,
			decoder_weights,
			kv_cache,
			hidden_states,
			causal_mask,
			position_ids,
			cache_position,
			layer_idx,
			sliding_window);

		hidden_states = layer_outputs;

		layer_idx++;
	}

	// calculate rms norm
	auto y = tensor_rms_norm(hidden_states, -1, model_weights.pos_rmsnorm_weight, config.rms_norm_eps, true);

	return y;
}

#endif // __GEMMA_MODEL__
