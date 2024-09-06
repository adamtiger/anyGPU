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
static SDPGradient<dtype, CPU> sdp_attention_bwd_cpu_precise_float(
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
static SDPGradient<dtype, CUDA> sdp_attention_bwd_cuda_basic(
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


/* qunatized forward implementation */

template<PreciseFloatType dtype>
static Tensor<int8, CPU> quant_sdp_attention_fwd_cpu_precise_i8(
	const Tensor<int8, CPU>& qw,
	const Tensor<int8, CPU>& kw,
	const Tensor<int8, CPU>& vw,
	const dtype sq, const int8 zpq,
	const dtype sk, const int8 zpk,
	const dtype sv, const int8 zpv,
	const dtype s1, const int8 zp1,
	const dtype s2, const int8 zp2,
	const dtype s3, const int8 zp3,
	const dtype sy, const int8 zpy)
{
	int d = qw.shape[1];  // qw shape: (N, d)
	dtype alpha = static_cast<dtype>(1.f / sqrtf(static_cast<float32>(d)));
	auto kw_tr = tensor_transp(kw);
	auto qk = tensor_qmm(qw, kw_tr, sq, zpq, sk, zpk, s1, zp1);
	auto deq_qk = tensor_dequantize_linear(qk, s2, zp2);
	auto nqk = tensor_mul(deq_qk, alpha);
	auto score = tensor_softmax(nqk);
	auto q_score = tensor_quantize_linear(score, s3, zp3);
	auto y = tensor_qmm(q_score, vw, s3, zp3, sv, zpv, sy, zpy);
	return y;
}



static Tensor<int8, CUDA> quant_sdp_attention_fwd_cuda_basic_f32_i8(
	const Tensor<int8, CUDA>& qw,
	const Tensor<int8, CUDA>& kw,
	const Tensor<int8, CUDA>& vw,
	const float32 sq, const int8 zpq,
	const float32 sk, const int8 zpk,
	const float32 sv, const int8 zpv,
	const float32 s1, const int8 zp1,
	const float32 s2, const int8 zp2,
	const float32 s3, const int8 zp3,
	const float32 sy, const int8 zpy)
{
	int d = qw.shape[1];  // qw shape: (N, d)
	float32 alpha = 1.f / sqrtf(static_cast<float32>(d));
	auto kw_tr = tensor_transp(kw);
	auto qk = tensor_qmm(qw, kw_tr, sq, zpq, sk, zpk, s1, zp1);
	auto deq_qk = tensor_dequantize_linear(qk, s2, zp2);
	auto nqk = tensor_mul(deq_qk, alpha);
	auto score = tensor_softmax(nqk);
	auto q_score = tensor_quantize_linear(score, s3, zp3);
	auto y = tensor_qmm(q_score, vw, s3, zp3, sv, zpv, sy, zpy);
	return y;
}

#endif  // __SDP__
