# DNN blocks for different GPUs

## Core, basics

- [x] create a core which can be used in .cu files (Tensor needs to be the input for some functions)
- [x] device query function
- [x] test files (skeleton for testing)
- [x] skeleton for attention - implement basic MM
- [x] skeleton for attention - implement division with tensor, scalar format
- [x] skeleton for attention - implement basic transpose (this will be eliminated later)
- [x] skeleton for attention - implement basic softmax
- [x] skeleton for attention - implement forward basic sdp
- [x] test by comparing to external - test case from torch
- [x] skeleton for attention - implement backward basic sdp
- [x] skeleton for attention - implement quantized forward basic sdp
    - [x] implement quantize linear
	- [x] implement dequantize linear
	- [x] implement quantized matmul
- [x] custom logger, assert, better error checks, small refactor (interface clarification)
    - [x] attention cuda - alpha should be calculated for fp16 and bf16 too
	- [x] alignment for cpu
	- [x] errors for tensor reading from file
	- [x] remove tranformer type, mask type and score function type from the template parameters
	- [x] test result comparison should be handled by separate functions
- [x] add fp8
- [x] tensor creator - full of with ones

- [x] optimization step1 for MM - shared memory (tiling)
- [x] fp16 conversion of mm kernel
- [x] cuda error checks (error 700 was hidden)
- [x] normal distrib tensor creator for floats

- [x] safe tensor reader (https://huggingface.co/docs/safetensors/index#format)
- [x] test for the safetensor reader


## Zambra model elements

- [x] zyphra model elements identification (https://github.com/Zyphra/Zamba2)
    - [x] clean python implementation (https://huggingface.co/Zyphra/Zamba2-1.2B)
	- [x] zamba2 folder

- [x] zamba2 basic model elements: (https://github.com/Zyphra/Zamba2/blob/main/mlp.py#L73)
    - [x] embedding layer (torch equivalent)
    - [x] implement linear (torch equivalent)
	- [x] layernorm (torch) (https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#layernorm)
	- [x] silu
	
	- [x] RMSNorm (https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.RMSNorm)
	- [x] RotaryEmbedding (original paper equivalent)
	- [x] RotaryEmbedding simplified (gemma2, zambra2 etc. applies different impl.)
	- [x] positionIds for rotary embedding 
	- [x] RotaryEmbedding for gpu
	(https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html on linux as banchmark)

- [x] naming should follow the same convention everywhere (cu_tensor_... in kernels)

- [x] create a python package for observing network blocks
    - tensor saver should be in it
	- logger like module to save the network block related info (config, input tensor, output tensor)
	- save the network info into a json file
	- (this is important, otherwise these functionalities will not be available in the transformers package)

- [slice in last dim](https://github.com/Zyphra/transformers_zamba2/blob/main/src/transformers/models/zamba2/modeling_zamba2.py#L783)
    - this is not used in zamba1.2 -> has no effect, can be ignored

- [x] zamba2 medium blocks (implementation)
    - link: https://github.com/Zyphra/transformers_zamba2/tree/main/src/transformers/models/zamba2
	- [x] implement [Zamba2RMSNorm](https://github.com/Zyphra/transformers_zamba2/blob/main/src/transformers/models/zamba2/modeling_zamba2.py#L99)
    - [x] implement gated linear unit
	- [x] implement [Zamba2MLP](https://github.com/Zyphra/transformers_zamba2/blob/main/src/transformers/models/zamba2/modeling_zamba2.py#L820)
	- [x] implement [Zamba2RotaryEmbedding](https://github.com/Zyphra/transformers_zamba2/blob/main/src/transformers/models/zamba2/modeling_zamba2.py#L116)
	- [x] implement Transpose_permutation (torch.transpose(1, 2) or similar, a full permutation support transpose)
	- [x] implement sdp attention with causal mask, and softmax scale (or with attention mask)

- [x] zamba2 attention decoder layer (implementation)
    - [x] implement concatenation on last dimension
    - [x] implement [Zamba2SdpaAttention](https://github.com/Zyphra/transformers_zamba2/blob/main/src/transformers/models/zamba2/modeling_zamba2.py#L707)
    - [x] implement [Zamba2AttentionDecoderLayer](https://github.com/Zyphra/transformers_zamba2/blob/main/src/transformers/models/zamba2/modeling_zamba2.py#L881)


## Gemma model baseline implementation

- [x] create gemma2 folder
    - for gemma2, almost everything is given
	- it is possible to move ahead and focus on:
		. quantization, pruning
		. optimization (memory, latency)
		. backward

- [x] gemma2 basic block implementation and tests
    - [transformers link](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma2)
	- [huggingface, gemma2-2b](https://huggingface.co/google/gemma-2b)
	- [x] implement gelu activation
	- [x] implement [mlp](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py#L71)
	- [x] test for rmsnorm (required a slight modification)
	- [x] test for rotary (same as zamba2)
	- [x] implement repeat_kv
	- [x] test sdpa
	- [x] implement slicing (last dime for attention mechanism, can be general with copy)
	- [x] implement [Gamma2DecoderLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py#L488)
    - [x] implement [Gemma2Model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py#L720)
        - [x] update causal mask needs to be implemented (target_length = 41)
		- [x] implement kv cache (hybrid) [link](https://github.com/Zyphra/transformers_zamba2/blob/main/src/transformers/cache_utils.py#L1132)
			. is the update static? (both, alternating)

- [x] gemma2 causal lm implementation
    - save all config params with inspector
    - contains a linear mapping at the end -> its weights are tied to the embedding weights (in gemma2 model)
	- it is not transposed -> needs a linear transformation that can handle this!
	
	- [x] implement lm_head combined with logit_softcapping (logit_softcapping param = 30.0)
		- num_logits_to_keep slicing before this calculation can speed up the process!


- generate implementation
    - [generate function](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1861)

- zambra sdpa can require different size calculations (see gemma2 for details!)

- [x] compilation on linux 
    - vulkan [linux install guide](https://vulkan.lunarg.com/doc/sdk/1.3.296.0/linux/getting_started_ubuntu.html)
	- latest cmake on ubuntu (sudo snap install cmake --classic)
	- problems:
	    . in template defitions, use inline instead of statis (explicit template specialization can not have a storage class)
		. templated member function copy_to_device does not work with gcc (but as a normal template function it can)
		. cuda headers are not found by cmake but it seems to found stuff!

- [x] error fix on linux
    - cuda maybe is not found properly -> vague error was received (what():  cannot create std::vector larger than max_size())
	- most likely this is due to unknown instructions, wrong versions etc.
	
	- driver version was not appropriate, better installation resolved the issue

- [x] vulkan has core dump error and complains about too long std::vector creation
    - only happens on linux, (AWS machine)
	- may be a return value is not fine??
	
	- shader was missing, wrong path were set for shader loading (std::vector<char> was the problem)
	- error handling will be necessary!


- [x] size should be changed to int64
- [x] introduce int64

- [x] test decoder with more io checkpoints
    . works with 30 checkpoints on any layer (even, odd)

- [x] num_logits_to_keep for speeding up some calculation in the last lm_head
    . use slicing for now (for performance, the typical values are needed!)

- [x] implement attention preprocessing in gemma2 decoder layer
    . [link](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py#L539)

- [x] full model tests and error fixes
    . for several iterations (will require linux at first time)
