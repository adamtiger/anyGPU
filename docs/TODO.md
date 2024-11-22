# DNN implementation with acceleration (on any gpus)

## Current steps to be done

- [ ] can copy_to_cuda dramatically increase the memory consumption?

- [ ] fp16, bfp16 kernel versions

- [ ] float to bfloat16 and other similar conversions can be done with static_cast (consider this)
- [ ] bf16 support for tensor saver (numpy cannot support bfloat16)


- [ ] examine the model calculation inaccuracy over time
    . 5% l2 error rate
	. where the difference arise (look for possible source in kv cache update and mask update)
	. (the real test will be the long sequence with generator)
	. optimization can also change the course of the divergence
	. overall the maximum points are the important


- [ ] cmake enhancement
    . cuda with linux
	. cuda for any machine (sm is hardcoded yet)
	. shader compilation with cmake?
