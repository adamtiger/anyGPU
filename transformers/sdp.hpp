#ifndef __SDP__
#define __SDP__

#include "ops.hpp"

#include "tensor.hpp"
#include "core_concepts.hpp"


/* forward implementation */

template<PreciseFloatType dtype>
Tensor<dtype, CPU> sdp_attention_fwd_cpu_precise_float(
	const Tensor<dtype, CPU>& qw,
	const Tensor<dtype, CPU>& kw,
	const Tensor<dtype, CPU>& vw)
{
	int d = qw.shape[1];  // qw shape: (N, d)
	dtype alpha = static_cast<dtype>(1.f / sqrtf(static_cast<float32>(d)));
	auto kw_tr = tensor_transp(kw);
	auto qk = tensor_mm(qw, kw_tr);
	auto nqk = tensor_mul(qk, alpha);
	auto score = tensor_softmax(nqk);
	auto y = tensor_mm(score, vw);
	return y;
}


template<FloatingPointType dtype>
Tensor<dtype, CUDA> sdp_attention_fwd_cuda_basic(
	const Tensor<dtype, CUDA>& qw,
	const Tensor<dtype, CUDA>& kw,
	const Tensor<dtype, CUDA>& vw)
{
	int d = qw.shape[1];  // qw shape: (N, d)
	dtype alpha = static_cast<dtype>(1.f / sqrtf(static_cast<float32>(d)));  // TODO: fp16 and bfp16
	auto kw_tr = tensor_transp(kw);
	auto qk = tensor_mm(qw, kw_tr);
	auto nqk = tensor_mul(qk, alpha);
	auto score = tensor_softmax(nqk);
	auto y = tensor_mm(score, vw);
	return y;
}



/*
   Returning sdp gradients is by a dedicated structure.
   This is slower but does not require outputs on the argument.
*/
template<typename dtype, Device device>
struct SDPGradient
{
	Tensor<dtype, device> grad_q;
	Tensor<dtype, device> grad_k;
	Tensor<dtype, device> grad_v;
};

/* backward implementation */

template<PreciseFloatType dtype>
SDPGradient<dtype, CPU> sdp_attention_bwd_cpu_precise_float(
	const Tensor<dtype, CPU>& qw,
	const Tensor<dtype, CPU>& kw,
	const Tensor<dtype, CPU>& vw)
{
	SDPGradient<dtype, CPU> grads;
	return grads;
}


template<FloatingPointType dtype>
SDPGradient<dtype, CUDA> sdp_attention_bwd_cuda_basic(
	const Tensor<dtype, CUDA>& qw,
	const Tensor<dtype, CUDA>& kw,
	const Tensor<dtype, CUDA>& vw)
{
	SDPGradient<dtype, CUDA> grads;
	return grads;
}

#endif  // __SDP__
