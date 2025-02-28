#ifndef __ATTENTION__
#define __ATTENTION__

#include <optional>
#include "tensor.hpp"
#include "core_concepts.hpp"

#include "sdp.hpp"

/*
*  This is the api header for the attention
*  implementation. 
*/

/*
*  The attention mechanism has different
*  variants. The main types are differentiated
*  by how the query, key and value matrices
*  are used.
*/
enum TransformerKind
{
	SDP,  // single-dot-product
	MHA,  // multi-head attention
	MQA,  // multi-query attention
	GQA   // group-query attention
};


/*
*  Masking can have also different types.
*/
enum MaskKind
{
	NONE,
	CAUSAL,
	SLIDING_WIN
};


/*
*  Score functions applied right before softmax.
*/
enum ScoreFuncKind
{
	FULL,  // no score function
	SOFT_CAPPING
};


/*
*  Top-level attention implementations.
*/


/*
*  Forward single head attention implementations.
*  The different type of transformers can be
*  specified with the template parameters.
*/
template<FloatingPointType dtype, Device device>
inline Tensor<dtype, device> single_head_attention_fwd(
	const Tensor<dtype, device>& qw,
	const Tensor<dtype, device>& kw,
	const Tensor<dtype, device>& vw) 
{
	Tensor<dtype, device> y;
	return y;
};

/* cpu implementations */

template<>
inline Tensor<float32, CPU> single_head_attention_fwd<float32, CPU>(
	const Tensor<float32, CPU>& qw,
	const Tensor<float32, CPU>& kw,
	const Tensor<float32, CPU>& vw)
{
	return sdp_attention_fwd_cpu_precise_float(qw, kw, vw);
}

template<>
inline Tensor<float64, CPU> single_head_attention_fwd<float64, CPU>(
	const Tensor<float64, CPU>& qw,
	const Tensor<float64, CPU>& kw,
	const Tensor<float64, CPU>& vw)
{
	return sdp_attention_fwd_cpu_precise_float(qw, kw, vw);
}

/* not implemented cases for cpu (floats under 4 bytes) */

template<>
inline Tensor<float16, CPU> single_head_attention_fwd<float16, CPU>(
	const Tensor<float16, CPU>& qw,
	const Tensor<float16, CPU>& kw,
	const Tensor<float16, CPU>& vw) = delete;

template<>
inline Tensor<bfloat16, CPU> single_head_attention_fwd<bfloat16, CPU>(
	const Tensor<bfloat16, CPU>& qw,
	const Tensor<bfloat16, CPU>& kw,
	const Tensor<bfloat16, CPU>& vw) = delete;

/* cuda implementations */
template<>
inline Tensor<float32, CUDA> single_head_attention_fwd<float32, CUDA>(
	const Tensor<float32, CUDA>& qw,
	const Tensor<float32, CUDA>& kw,
	const Tensor<float32, CUDA>& vw)
{
	return sdp_attention_fwd_cuda_flash(qw, kw, vw);
}





/*
*  Backward single head attention implementations.
*  The different type of transformers can be
*  specified with the template parameters.
*/
template<FloatingPointType dtype, Device device>
inline SDPGradient<dtype, device> single_head_attention_bwd(
	const Tensor<dtype, device>& qw,
	const Tensor<dtype, device>& kw,
	const Tensor<dtype, device>& vw,
	const Tensor<dtype, device>& grad_y)
{
	SDPGradient<dtype, device> grads;
	return grads;
};

/* cpu implementations */

template<>
inline SDPGradient<float32, CPU> single_head_attention_bwd<float32, CPU>(
	const Tensor<float32, CPU>& qw,
	const Tensor<float32, CPU>& kw,
	const Tensor<float32, CPU>& vw,
	const Tensor<float32, CPU>& grad_y)
{
	return sdp_attention_bwd_cpu_precise_float(qw, kw, vw, grad_y);
}

template<>
inline SDPGradient<float64, CPU> single_head_attention_bwd<float64, CPU>(
	const Tensor<float64, CPU>& qw,
	const Tensor<float64, CPU>& kw,
	const Tensor<float64, CPU>& vw,
	const Tensor<float64, CPU>& grad_y)
{
	return sdp_attention_bwd_cpu_precise_float(qw, kw, vw, grad_y);
}

/* not implemented cases for cpu (floats under 4 bytes) */

template<>
inline SDPGradient<float16, CPU> single_head_attention_bwd<float16, CPU>(
	const Tensor<float16, CPU>& qw,
	const Tensor<float16, CPU>& kw,
	const Tensor<float16, CPU>& vw,
	const Tensor<float16, CPU>& grad_y) = delete;

template<>
inline SDPGradient<bfloat16, CPU> single_head_attention_bwd<bfloat16, CPU>(
	const Tensor<bfloat16, CPU>& qw,
	const Tensor<bfloat16, CPU>& kw,
	const Tensor<bfloat16, CPU>& vw,
	const Tensor<bfloat16, CPU>& grad_y) = delete;

/* cuda implementations */
template<>
inline SDPGradient<float32, CUDA> single_head_attention_bwd<float32, CUDA>(
	const Tensor<float32, CUDA>& qw,
	const Tensor<float32, CUDA>& kw,
	const Tensor<float32, CUDA>& vw,
	const Tensor<float32, CUDA>& grad_y)
{
	return sdp_attention_bwd_cuda_basic(qw, kw, vw, grad_y);
}





/*
*  Quantized forward single head attention implementations.
*  The different type of transformers can be
*  specified with the template parameters.
*  @param s1: scale in the quantization of R1 (the result of first matmul)
*  @param zp1: zeropoint in the quantization of R1 (the result of first matmul)
*  @param s2: scale in the dequantize linear before alpha mul.
*  @param zp2: zeropoint in the dequantize linear before alpha mul.
*  @param s3: scale in the quantize linear after softmax
*  @param zp3: zeropoint in the quantize linear after softmax
*/
template<
	FloatingPointType hp_dtype,
	IntegerType lp_dtype,
	Device device>
inline Tensor<lp_dtype, device> quantized_single_head_attention_fwd(
	const Tensor<lp_dtype, device>& qw,
	const Tensor<lp_dtype, device>& kw,
	const Tensor<lp_dtype, device>& vw,
	const hp_dtype sq, const lp_dtype zpq,
	const hp_dtype sk, const lp_dtype zpk,
	const hp_dtype sv, const lp_dtype zpv,
	const hp_dtype s1, const lp_dtype zp1,
	const hp_dtype s2, const lp_dtype zp2,
	const hp_dtype s3, const lp_dtype zp3,
	const hp_dtype sy, const lp_dtype zpy)
{
	Tensor<lp_dtype, device> y;
	return y;
};

/* cpu implementations */

template<>
inline Tensor<int8, CPU> quantized_single_head_attention_fwd<float32, int8, CPU>(
	const Tensor<int8, CPU>& qw,
	const Tensor<int8, CPU>& kw,
	const Tensor<int8, CPU>& vw,
	const float32 sq, const int8 zpq,
	const float32 sk, const int8 zpk,
	const float32 sv, const int8 zpv,
	const float32 s1, const int8 zp1,
	const float32 s2, const int8 zp2,
	const float32 s3, const int8 zp3,
	const float32 sy, const int8 zpy)
{
	return quant_sdp_attention_fwd_cpu_precise_i8(
		qw, kw, vw,
		sq, zpq,
		sk, zpk,
		sv, zpv,
		s1, zp1,
		s2, zp2,
		s3, zp3,
		sy, zpy
	);
}

template<>
inline Tensor<int8, CPU> quantized_single_head_attention_fwd<float64, int8, CPU>(
	const Tensor<int8, CPU>& qw,
	const Tensor<int8, CPU>& kw,
	const Tensor<int8, CPU>& vw,
	const float64 sq, const int8 zpq,
	const float64 sk, const int8 zpk,
	const float64 sv, const int8 zpv,
	const float64 s1, const int8 zp1,
	const float64 s2, const int8 zp2,
	const float64 s3, const int8 zp3,
	const float64 sy, const int8 zpy)
{
	return quant_sdp_attention_fwd_cpu_precise_i8(
		qw, kw, vw,
		sq, zpq,
		sk, zpk,
		sv, zpv,
		s1, zp1,
		s2, zp2,
		s3, zp3,
		sy, zpy
	);
}

/* not implemented cases for cpu (floats under 4 bytes) */

template<>
inline Tensor<int8, CPU> quantized_single_head_attention_fwd<float16, int8, CPU>(
	const Tensor<int8, CPU>& qw,
	const Tensor<int8, CPU>& kw,
	const Tensor<int8, CPU>& vw,
	const float16 sq, const int8 zpq,
	const float16 sk, const int8 zpk,
	const float16 sv, const int8 zpv,
	const float16 s1, const int8 zp1,
	const float16 s2, const int8 zp2,
	const float16 s3, const int8 zp3,
	const float16 sy, const int8 zpy) = delete;

template<>
inline Tensor<int8, CPU> quantized_single_head_attention_fwd<bfloat16, int8, CPU>(
	const Tensor<int8, CPU>& qw,
	const Tensor<int8, CPU>& kw,
	const Tensor<int8, CPU>& vw,
	const bfloat16 sq, const int8 zpq,
	const bfloat16 sk, const int8 zpk,
	const bfloat16 sv, const int8 zpv,
	const bfloat16 s1, const int8 zp1,
	const bfloat16 s2, const int8 zp2,
	const bfloat16 s3, const int8 zp3,
	const bfloat16 sy, const int8 zpy) = delete;

/* cuda implementations */
template<>
inline Tensor<int8, CUDA> quantized_single_head_attention_fwd<float32, int8, CUDA>(
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
	return quant_sdp_attention_fwd_cuda_basic_f32_i8(
		qw, kw, vw,
		sq, zpq,
		sk, zpk,
		sv, zpv,
		s1, zp1,
		s2, zp2,
		s3, zp3,
		sy, zpy
	);
}

#endif  // __ATTENTION__
