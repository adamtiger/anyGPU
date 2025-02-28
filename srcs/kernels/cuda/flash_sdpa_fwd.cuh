#ifndef __FLASH_SDPA_FWD_CUH__
#define __FLASH_SDPA_FWD_CUH__

#include "tensor.hpp"

// baseline implementation
// hidden dim: 256
// seq length: expected to be a multiple of 132
// assumed max shared mem: 49152(byte)
void cu_flash_sdpa_fwd_d256_v1(
	const float32 alpha,
	const Tensor<float32, CUDA>& qt,
	const Tensor<float32, CUDA>& kt,
	const Tensor<float32, CUDA>& vt,
	Tensor<float32, CUDA>& yt);


#endif  // __FLASH_SDPA_FWD_CUH__
