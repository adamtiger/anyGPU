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

	GemmaConfig() :
		head_dim(-1),
		hidden_size(-1),
		num_attention_heads(-1),
		num_key_value_heads(-1),
		num_hidden_layers(-1),
		sliding_window(-1)
	{
	}
};

#endif  //__GEMMA_CONFIG__
