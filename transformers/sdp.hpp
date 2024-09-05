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
	const Tensor<dtype, CPU>& vw,
	const Tensor<dtype, CPU>& grad_y)
{
	SDPGradient<dtype, CPU> grads;

	// forward recompute to calculate the softmax result
	int d = qw.shape[1];  // qw shape: (N, d)
	dtype alpha = static_cast<dtype>(1.f / sqrtf(static_cast<float32>(d)));
	auto kw_tr = tensor_transp(kw);
	auto qk = tensor_mm(qw, kw_tr);
	auto nqk = tensor_mul(qk, alpha);
	auto r3 = tensor_softmax(nqk);

	// calculate the backward steps
	grads.grad_v = tensor_mm(tensor_transp(r3), grad_y);
	auto dr3 = tensor_mm(grad_y, tensor_transp(vw));
	auto dr2 = tensor_softmax_bwd(r3, dr3);
	auto dr1 = tensor_mul(dr2, alpha);
	grads.grad_q = tensor_mm(dr1, kw);
	auto dr0 = tensor_mm(tensor_transp(qw), dr1);
	grads.grad_k = tensor_transp(dr0);

	return grads;
}


template<FloatingPointType dtype>
SDPGradient<dtype, CUDA> sdp_attention_bwd_cuda_basic(
	const Tensor<dtype, CUDA>& qw,
	const Tensor<dtype, CUDA>& kw,
	const Tensor<dtype, CUDA>& vw,
	const Tensor<dtype, CUDA>& grad_y)
{
	SDPGradient<dtype, CUDA> grads;

	// forward recompute to calculate the softmax result
	int d = qw.shape[1];  // qw shape: (N, d)
	dtype alpha = static_cast<dtype>(1.f / sqrtf(static_cast<float32>(d)));
	auto kw_tr = tensor_transp(kw);
	auto qk = tensor_mm(qw, kw_tr);
	auto nqk = tensor_mul(qk, alpha);
	auto r3 = tensor_softmax(nqk);

	// calculate the backward steps
	grads.grad_v = tensor_mm(tensor_transp(r3), grad_y);
	auto dr3 = tensor_mm(grad_y, tensor_transp(vw));
	auto dr2 = tensor_softmax_bwd(r3, dr3);
	auto dr1 = tensor_mul(dr2, alpha);
	grads.grad_q = tensor_mm(dr1, kw);
	auto dr0 = tensor_mm(tensor_transp(qw), dr1);
	grads.grad_k = tensor_transp(dr0);

	return grads;
}

#endif  // __SDP__
