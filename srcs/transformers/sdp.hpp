#ifndef __SDP__
#define __SDP__

#include "flash_sdpa_fwd.cuh"
#include "ops.hpp"

#include "tensor.hpp"
#include "core_concepts.hpp"
#include "attention_tools.hpp"


/* forward implementation */

template<PreciseFloatType dtype>
inline Tensor<dtype, CPU> sdp_attention_fwd_cpu_precise_float(
	const Tensor<dtype, CPU>& qw,
	const Tensor<dtype, CPU>& kw,
	const Tensor<dtype, CPU>& vw)
{
	int d = qw.shape[1];  // qw shape: (N, d)
	dtype alpha = calculate_alpha<dtype>(d);
	auto kw_tr = tensor_transp(kw);
	auto qk = tensor_mm(qw, kw_tr);
	auto nqk = tensor_mul(qk, alpha);
	auto score = tensor_softmax(nqk);
	auto y = tensor_mm(score, vw);
	return y;
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> sdp_attention_fwd_cuda_basic(
	const Tensor<dtype, CUDA>& qw,
	const Tensor<dtype, CUDA>& kw,
	const Tensor<dtype, CUDA>& vw)
{
	int d = qw.shape[1];  // qw shape: (N, d)
	dtype alpha = calculate_alpha<dtype>(d);
	auto kw_tr = tensor_transp(kw);
	auto qk = tensor_mm(qw, kw_tr);
	auto nqk = tensor_mul(qk, alpha);
	auto score = tensor_softmax(nqk);
	auto y = tensor_mm(score, vw);
	return y;
}


template<FloatingPointType dtype>
inline Tensor<dtype, CUDA> sdp_attention_fwd_cuda_flash(
	const Tensor<dtype, CUDA>& qw,
	const Tensor<dtype, CUDA>& kw,
	const Tensor<dtype, CUDA>& vw)
{
	int N = qw.shape[0];
	int d = qw.shape[1];  // qw shape: (N, d)

	ACASSERT(kw.shape[0] == N, "qw and kw differs in sequence length");
	ACASSERT(vw.shape[0] == N, "qw and vw differs in sequence length");
	ACASSERT(qw.shape[1] == 256, "hidden dimension for kw must be 256");
	ACASSERT(kw.shape[1] == 256, "hidden dimension for kw must be 256");
	ACASSERT(vw.shape[1] == 256, "hidden dimension for kw must be 256");

	Tensor<dtype, CUDA> y(qw.dim, qw.shape);

	dtype alpha = calculate_alpha<dtype>(d);
	
	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_flash_sdpa_fwd_d256_v1(static_cast<float32>(alpha), qw, kw, vw, y);
	}

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
inline SDPGradient<dtype, CPU> sdp_attention_bwd_cpu_precise_float(
	const Tensor<dtype, CPU>& qw,
	const Tensor<dtype, CPU>& kw,
	const Tensor<dtype, CPU>& vw,
	const Tensor<dtype, CPU>& grad_y)
{
	SDPGradient<dtype, CPU> grads;

	// forward recompute to calculate the softmax result
	int d = qw.shape[1];  // qw shape: (N, d)
	dtype alpha = calculate_alpha<dtype>(d);
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
inline SDPGradient<dtype, CUDA> sdp_attention_bwd_cuda_basic(
	const Tensor<dtype, CUDA>& qw,
	const Tensor<dtype, CUDA>& kw,
	const Tensor<dtype, CUDA>& vw,
	const Tensor<dtype, CUDA>& grad_y)
{
	SDPGradient<dtype, CUDA> grads;

	// forward recompute to calculate the softmax result
	int d = qw.shape[1];  // qw shape: (N, d)
	dtype alpha = calculate_alpha<dtype>(d);
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
inline Tensor<int8, CPU> quant_sdp_attention_fwd_cpu_precise_i8(
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
	dtype alpha = calculate_alpha<dtype>(d);
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
	float32 alpha = calculate_alpha<float32>(d);
	auto kw_tr = tensor_transp(kw);
	auto qk = tensor_qmm(qw, kw_tr, sq, zpq, sk, zpk, s1, zp1);
	auto deq_qk = tensor_dequantize_linear(qk, s2, zp2);
	auto nqk = tensor_mul(deq_qk, alpha);
	auto score = tensor_softmax(nqk);
	auto q_score = tensor_quantize_linear(score, s3, zp3);
	auto y = tensor_qmm(q_score, vw, s3, zp3, sv, zpv, sy, zpy);
	return y;
}



/* SDP with masking and scaling, shape is general */

template<FloatingPointType dtype, Device device>
inline Tensor<dtype, device> sdp_attention_masked_scaled_fwd(
	const Tensor<dtype, device>& qw,
	const Tensor<dtype, device>& kw,
	const Tensor<dtype, device>& vw,
	const Tensor<dtype, device>& mask,
	const dtype alpha)
{
	if constexpr (device == CPU)
	{
		static_assert(PreciseFloatType<dtype>, "for cpu only precise float is supported");
	}

	// qw, kw, vw are not 2 dimensional but instead 3 or 4
	// qw, kw, vw are assumed to have similar shapes
	// mask is assumed to have 1s in the first dims
	ACASSERT(qw.dim == kw.dim && vw.dim == kw.dim, "dimension missmatch");
	ACASSERT(mask.dim == qw.dim, "mask dimension is inappropriate");

	// calculate the number of 2d matrices
	int dim = qw.dim;
	int num_mtcs = 1;
	for (int ix = 0; ix < dim - 2; ++ix)
	{
		num_mtcs *= qw.shape[ix];

		ACASSERT(
			qw.shape[ix] == kw.shape[ix] && kw.shape[ix] == vw.shape[ix], 
			"qw, kw, vw needs to have the same number of 2d matrices"
		);

		ACASSERT(mask.shape[ix] == 1, "mask needs to have 1s in the first dims except the last two");
	}

	// view, with 3d
	auto qwn = tensor_view(qw, { num_mtcs, qw.shape[dim - 2], qw.shape[dim - 1] });
	auto kwn = tensor_view(kw, { num_mtcs, kw.shape[dim - 2], kw.shape[dim - 1] });
	auto vwn = tensor_view(vw, { num_mtcs, vw.shape[dim - 2], vw.shape[dim - 1] });
	auto maskn = tensor_view(mask, { mask.shape[dim - 2], mask.shape[dim - 1] });

	// split (no memory copy)
	auto qw_mtcs = tensor_split(qwn, num_mtcs);
	auto kw_mtcs = tensor_split(kwn, num_mtcs);
	auto vw_mtcs = tensor_split(vwn, num_mtcs);

	// output
	Tensor<dtype, device> y(dim, qw.shape);
	auto yn = tensor_view(y, { num_mtcs, y.shape[dim - 2], y.shape[dim - 1] });
	auto y_mtcs = tensor_split(yn, num_mtcs);

	// calculate attention for each submatrix
	for (int ix = 0; ix < num_mtcs; ++ix)
	{
		auto qw_mtx = tensor_view(qw_mtcs[ix], { qw_mtcs[ix].shape[1], qw_mtcs[ix].shape[2] });
		auto kw_mtx = tensor_view(kw_mtcs[ix], { kw_mtcs[ix].shape[1], kw_mtcs[ix].shape[2] });
		auto vw_mtx = tensor_view(vw_mtcs[ix], { vw_mtcs[ix].shape[1], vw_mtcs[ix].shape[2] });
		auto y_mtx = tensor_view(y_mtcs[ix], { y_mtcs[ix].shape[1], y_mtcs[ix].shape[2] });

		auto kw_tr = tensor_transp(kw_mtx);
		auto qk = tensor_mm(qw_mtx, kw_tr);
		auto nqk = tensor_mul(qk, alpha);
		auto nqk_masked = tensor_add(nqk, maskn);
		auto score = tensor_softmax(nqk_masked);
		tensor_mm(score, vw_mtx, y_mtx);
	}

	return y;
}


#endif  // __SDP__
