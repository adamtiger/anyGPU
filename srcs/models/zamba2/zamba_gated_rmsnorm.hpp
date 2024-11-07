#ifndef __ZAMBA_GATED_RMSNORM__
#define __ZAMBA_GATED_RMSNORM__

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "ops.hpp"


template<FloatingPointType dtype, Device device>
inline Tensor<dtype, device> tensor_zamba_gated_rmsnorm(
	const Tensor<dtype, device>& xt,
	const Tensor<dtype, device>& zt,
	const Tensor<dtype, device>& wt,
	const int gsize,
	const dtype eps)
{
	// TODO: the gated mechanism is not implemented here but zamba case seems special,
	//       it is a fallback to the normal case
	auto silu_zt = tensor_silu(zt);
	auto gated = tensor_mul(xt, silu_zt);
	auto yt = tensor_rms_norm(gated, -1, wt, eps);
	return yt;
}

#endif  // __ZAMBA_GATED_RMSNORM__
