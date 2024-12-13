# DNN implementation with acceleration (on any gpus)

## Current steps to be done



### Optimization

- [x] measure the execution time per operator
- [x] create a bar plot about exec times
- [x] create separate folder for cuda kernels

- [x] optimization step: mlp
- [x] check the fused mlp kernel on more tests
- [x] optimization step: mlp linear part

- [x] check performance distribution inside sdpa
- [x] sdpa linear functions, need optim
- [x] test performance with seq_len 1, token generation case (does the bottleneck change? -> not much)
- [x] test examples for fused mlp with positive and uniform random values

- [ ] float to bfloat16 and other similar conversions can be done with static_cast (consider this)
- [ ] bf16 support for tensor saver (numpy cannot support bfloat16)


- quantization
- sparse implementation
- other techniques ...


### Memory and data

- [x] read model weights directly from safetensor (instead of dat)

- [ ] arena implementation
    - [ ] default arena (falls back to cudaMalloc, dynamic memory allocation)
	- [ ] record the allocation pattern in the graph (default allocator with monitoring)
	- [ ] allocate the arena once, create tensors from the arena (no cost)
	    - this requires to calculate an allocation strategy (can be optimized because it is known in advance)
	- ...


### Build infrastructure

- [x] remove vulkan (temporary it is in another repo)
- [ ] cmake enhancement
    . cuda with linux
	. cuda for any machine (sm is hardcoded yet)
	. shader compilation with cmake?


### Questions (needs research)

- [ ] can copy_to_cuda dramatically increase the memory consumption?


- save script should handle dictionary inputs
- safetensor io should be able to handle chunked weight lists (more saftensors for a single model) 
    - e.g. gemma model's safetensor loader would be more efficient (no vector resizes etc.)
