#ifndef __ATTENTION__
#define __ATTENTION__

#include <optional>
#include "tensor.hpp"

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

template<
	DataType dtype, 
	TransformerKind trf_kind,
	MaskKind mask_kind,
	ScoreFuncKind sc_fn_kind>
static void attention_fwd(
	Tensor<dtype>& y,
	const Tensor<dtype>& x, 
	const Tensor<dtype>& qw, 
	const Tensor<dtype>& kw, 
	const Tensor<dtype>& vw,
	const std::optional<Tensor<dtype>>& ow = std::nullopt)
{

}


#endif  // __ATTENTION__
