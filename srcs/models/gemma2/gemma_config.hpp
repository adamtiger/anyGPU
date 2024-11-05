#ifndef __GEMMA_CONFIG__
#define __GEMMA_CONFIG__

#include "core.hpp"


struct GemmaConfig
{
	int32 head_dim;
	int32 hidden_size;
	int32 num_attention_heads;
	int32 num_key_value_heads;
	int32 num_hidden_layers;

	int32 sliding_window;
	int32 target_length;

	int32 rope_base;
	float32 rms_norm_eps;
	float32 sfmx_scale;
	float32 final_softcapping;

	GemmaConfig() :
		head_dim(256),
		hidden_size(2304),
		num_attention_heads(8),
		num_key_value_heads(4),
		num_hidden_layers(26),
		sliding_window(4096),
		target_length(41),
		rope_base(10000),
		rms_norm_eps(1e-6f),
		sfmx_scale(0.0625f),
		final_softcapping(30.f)
	{
	}

	void reset()
	{
		head_dim = -1;
		hidden_size = -1;
		num_attention_heads = -1;
		num_key_value_heads = -1;
		num_hidden_layers = -1;
		sliding_window = -1;
		target_length = -1;

		rope_base = 10000;     // typical default
		rms_norm_eps = 1e-6f;  // typical default
		sfmx_scale = 1.f;
		final_softcapping = 1.f;
	}
};

#endif  //__GEMMA_CONFIG__
