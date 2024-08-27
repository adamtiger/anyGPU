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
template<
	FloatingPointType dtype, 
	Device device,
	MaskKind mask_kind,
	ScoreFuncKind sc_fn_kind>
static Tensor<dtype, device> single_head_attention_fwd(
	const Tensor<dtype, device>& qw,
	const Tensor<dtype, device>& kw,
	const Tensor<dtype, device>& vw) 
{
	Tensor<dtype, device> y;
	return y;
};

/* cpu implementations */

template<>
static Tensor<float32, CPU> single_head_attention_fwd<float32, CPU, NONE, FULL>(
	const Tensor<float32, CPU>& qw,
	const Tensor<float32, CPU>& kw,
	const Tensor<float32, CPU>& vw)
{
	return sdp_attention_fwd_cpu_precise_float(qw, kw, vw);
}

template<>
static Tensor<float64, CPU> single_head_attention_fwd<float64, CPU, NONE, FULL>(
	const Tensor<float64, CPU>& qw,
	const Tensor<float64, CPU>& kw,
	const Tensor<float64, CPU>& vw)
{
	return sdp_attention_fwd_cpu_precise_float(qw, kw, vw);
}

/* not implemented cases for cpu (floats under 4 bytes) */

template<>
static Tensor<float16, CPU> single_head_attention_fwd<float16, CPU, NONE, FULL>(
	const Tensor<float16, CPU>& qw,
	const Tensor<float16, CPU>& kw,
	const Tensor<float16, CPU>& vw) = delete;

template<>
static Tensor<float16, CPU> single_head_attention_fwd<float16, CPU, NONE, SOFT_CAPPING>(
	const Tensor<float16, CPU>& qw,
	const Tensor<float16, CPU>& kw,
	const Tensor<float16, CPU>& vw) = delete;

template<>
static Tensor<float16, CPU> single_head_attention_fwd<float16, CPU, CAUSAL, FULL>(
	const Tensor<float16, CPU>& qw,
	const Tensor<float16, CPU>& kw,
	const Tensor<float16, CPU>& vw) = delete;

template<>
static Tensor<float16, CPU> single_head_attention_fwd<float16, CPU, CAUSAL, SOFT_CAPPING>(
	const Tensor<float16, CPU>& qw,
	const Tensor<float16, CPU>& kw,
	const Tensor<float16, CPU>& vw) = delete;

template<>
static Tensor<bfloat16, CPU> single_head_attention_fwd<bfloat16, CPU, NONE, FULL>(
	const Tensor<bfloat16, CPU>& qw,
	const Tensor<bfloat16, CPU>& kw,
	const Tensor<bfloat16, CPU>& vw) = delete;

template<>
static Tensor<bfloat16, CPU> single_head_attention_fwd<bfloat16, CPU, NONE, SOFT_CAPPING>(
	const Tensor<bfloat16, CPU>& qw,
	const Tensor<bfloat16, CPU>& kw,
	const Tensor<bfloat16, CPU>& vw) = delete;

template<>
static Tensor<bfloat16, CPU> single_head_attention_fwd<bfloat16, CPU, CAUSAL, FULL>(
	const Tensor<bfloat16, CPU>& qw,
	const Tensor<bfloat16, CPU>& kw,
	const Tensor<bfloat16, CPU>& vw) = delete;

template<>
static Tensor<bfloat16, CPU> single_head_attention_fwd<bfloat16, CPU, CAUSAL, SOFT_CAPPING>(
	const Tensor<bfloat16, CPU>& qw,
	const Tensor<bfloat16, CPU>& kw,
	const Tensor<bfloat16, CPU>& vw) = delete;


/* cuda implementations */
template<>
static Tensor<float32, CUDA> single_head_attention_fwd<float32, CUDA, NONE, FULL>(
	const Tensor<float32, CUDA>& qw,
	const Tensor<float32, CUDA>& kw,
	const Tensor<float32, CUDA>& vw)
{
	return sdp_attention_fwd_cuda_basic(qw, kw, vw);
}





/*
*  Backward single head attention implementations.
*  The different type of transformers can be
*  specified with the template parameters.
*/
template<
	FloatingPointType dtype,
	Device device,
	MaskKind mask_kind,
	ScoreFuncKind sc_fn_kind>
static SDPGradient<dtype, device> single_head_attention_bwd(
	const Tensor<dtype, device>& qw,
	const Tensor<dtype, device>& kw,
	const Tensor<dtype, device>& vw)
{
	SDPGradient<dtype, device> grads;
	return grads;
};

/* cpu implementations */

template<>
static SDPGradient<float32, CPU> single_head_attention_bwd<float32, CPU, NONE, FULL>(
	const Tensor<float32, CPU>& qw,
	const Tensor<float32, CPU>& kw,
	const Tensor<float32, CPU>& vw)
{
	return sdp_attention_bwd_cpu_precise_float(qw, kw, vw);
}

template<>
static SDPGradient<float64, CPU> single_head_attention_bwd<float64, CPU, NONE, FULL>(
	const Tensor<float64, CPU>& qw,
	const Tensor<float64, CPU>& kw,
	const Tensor<float64, CPU>& vw)
{
	return sdp_attention_bwd_cpu_precise_float(qw, kw, vw);
}

/* not implemented cases for cpu (floats under 4 bytes) */

template<>
static SDPGradient<float16, CPU> single_head_attention_bwd<float16, CPU, NONE, FULL>(
	const Tensor<float16, CPU>& qw,
	const Tensor<float16, CPU>& kw,
	const Tensor<float16, CPU>& vw) = delete;

template<>
static SDPGradient<float16, CPU> single_head_attention_bwd<float16, CPU, NONE, SOFT_CAPPING>(
	const Tensor<float16, CPU>& qw,
	const Tensor<float16, CPU>& kw,
	const Tensor<float16, CPU>& vw) = delete;

template<>
static SDPGradient<float16, CPU> single_head_attention_bwd<float16, CPU, CAUSAL, FULL>(
	const Tensor<float16, CPU>& qw,
	const Tensor<float16, CPU>& kw,
	const Tensor<float16, CPU>& vw) = delete;

template<>
static SDPGradient<float16, CPU> single_head_attention_bwd<float16, CPU, CAUSAL, SOFT_CAPPING>(
	const Tensor<float16, CPU>& qw,
	const Tensor<float16, CPU>& kw,
	const Tensor<float16, CPU>& vw) = delete;

template<>
static SDPGradient<bfloat16, CPU> single_head_attention_bwd<bfloat16, CPU, NONE, FULL>(
	const Tensor<bfloat16, CPU>& qw,
	const Tensor<bfloat16, CPU>& kw,
	const Tensor<bfloat16, CPU>& vw) = delete;

template<>
static SDPGradient<bfloat16, CPU> single_head_attention_bwd<bfloat16, CPU, NONE, SOFT_CAPPING>(
	const Tensor<bfloat16, CPU>& qw,
	const Tensor<bfloat16, CPU>& kw,
	const Tensor<bfloat16, CPU>& vw) = delete;

template<>
static SDPGradient<bfloat16, CPU> single_head_attention_bwd<bfloat16, CPU, CAUSAL, FULL>(
	const Tensor<bfloat16, CPU>& qw,
	const Tensor<bfloat16, CPU>& kw,
	const Tensor<bfloat16, CPU>& vw) = delete;

template<>
static SDPGradient<bfloat16, CPU> single_head_attention_bwd<bfloat16, CPU, CAUSAL, SOFT_CAPPING>(
	const Tensor<bfloat16, CPU>& qw,
	const Tensor<bfloat16, CPU>& kw,
	const Tensor<bfloat16, CPU>& vw) = delete;


/* cuda implementations */
template<>
static SDPGradient<float32, CUDA> single_head_attention_bwd<float32, CUDA, NONE, FULL>(
	const Tensor<float32, CUDA>& qw,
	const Tensor<float32, CUDA>& kw,
	const Tensor<float32, CUDA>& vw)
{
	return sdp_attention_bwd_cuda_basic(qw, kw, vw);
}

#endif  // __ATTENTION__
