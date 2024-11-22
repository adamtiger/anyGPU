# DNN implementation with acceleration (on any gpus)

## Current steps to be done


### Cleaning, adjustment

- [ ] copy test data folders to s3 bucket
- [ ] revisit the gemma2 implementation, delete unused parts
- [ ] identify possible missing test cases (uncovered functions) 
- [ ] execute more tests
- [ ] examine the model calculation inaccuracy over time
    . 5% l2 error rate
	. where the difference arise (look for possible source in kv cache update and mask update)
	. (the real test will be the long sequence with generator)
	. optimization can also change the course of the divergence
	. overall the maximum points are the important


### Memory and data

- [ ] read model weights directly from safetensor (instead of dat)

- [ ] arena implementation
    - [ ] default arena (falls back to cudaMalloc, dynamic memory allocation)
	- [ ] record the allocation pattern in the graph (default allocator with monitoring)
	- [ ] allocate the arena once, create tensors from the arena (no cost)
	    - this requires to calculate an allocation strategy (can be optimized because it is known in advance)
	- ...


### Optimization

- [ ] measure the execution time per operator
- [ ] create a bar plot about exec times

- [ ] fast f32 matmul for cuda

- [ ] fp16, bfp16 kernel versions

- [ ] float to bfloat16 and other similar conversions can be done with static_cast (consider this)
- [ ] bf16 support for tensor saver (numpy cannot support bfloat16)


- quantization
- sparse implementation
- other techniques ...


### Build infrastructure

- [x] remove vulkan (temporary it is in another repo)
- [ ] cmake enhancement
    . cuda with linux
	. cuda for any machine (sm is hardcoded yet)
	. shader compilation with cmake?


### Questions (needs research)

- [ ] can copy_to_cuda dramatically increase the memory consumption?







