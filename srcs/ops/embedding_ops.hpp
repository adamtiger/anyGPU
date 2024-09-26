#ifndef __EMBEDDING_OPS__
#define __EMBEDDING_OPS__

#include "embedding_ops.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


/*
  Embedding operator.
  @param xt: input tensor with indices
  @param wt: embedding weights
*/
template<PreciseFloatType dtype>
static Tensor<dtype, CPU> tensor_embedding(const Tensor<int32, CPU>& xt, const Tensor<dtype, CPU>& wt)
{
	// access the data arrays
	const int length = xt.size();
	const int emb_size = wt.shape[1];

	Shape y_shape = xt.shape;
	y_shape[xt.dim] = emb_size;
	Tensor<dtype, CPU> yt(xt.dim + 1, y_shape);
	dtype* y_data = yt.buffer();
	int32* x_data = xt.buffer();
	dtype* w_data = wt.buffer();

	// reference implementation
	// reliable (but slow)

	for (int k = 0; k < length; ++k)
	{
		int32 i = x_data[k];
		memcpy(y_data + k * emb_size, w_data + i * emb_size, emb_size * sizeof(dtype));
	}

	return yt;
}


template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_embedding(const Tensor<int32, CUDA>& xt, const Tensor<dtype, CUDA>& wt)
{
	// access the data arrays
	Shape y_shape = xt.shape;
	y_shape[xt.dim] = wt.shape[1];
	Tensor<dtype, CUDA> yt(xt.dim + 1, y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_embedding_f32(xt, wt, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}





/*
  Rotary position embedding operator.
  Precomputes the embedding tensor.
  Article compatible version (https://arxiv.org/pdf/2104.09864)
  @param max_seq_len: the maximum length of the token sequence
  @param emb_size: the size (length) of the token embedding vector
  @param base: the base for the geometric progression used to compute the rotation angles
*/
template<PreciseFloatType dtype>
static Tensor<dtype, CPU> tensor_precomp_rotary_embedding_cpu(
	const int32 max_seq_len, 
	const int32 emb_size, 
	const int32 base=10000)
{
	ACASSERT(emb_size % 2 == 0, "embedding size should be even");
	int32 d_per_2 = emb_size / 2;
	// initiate the frequence matrix
	Tensor<dtype, CPU> freq(2, { max_seq_len, d_per_2 });
	dtype* freq_data = freq.buffer();

	// calculate the base frequences for the embedding dimension
	std::vector<dtype> base_freq(d_per_2);

	dtype fi = 0.f;
	dtype fdper2 = static_cast<dtype>(d_per_2);
	dtype fbase = static_cast<dtype>(base);
	for (int32 i = 0; i < d_per_2; ++i)
	{
		dtype p = fi / fdper2;
		dtype theta = pow(fbase, -p);
		fi += 1.f;

		base_freq[i] = theta;
	}

	// calculate the outer product on the fly
	for (int32 m = 0; m < max_seq_len; ++m)
	{
		int32 base_offset = m * d_per_2;
		for (int32 i = 0; i < d_per_2; ++i)
		{
			freq_data[base_offset + i] = m * base_freq[i];
		}
	}

	return freq;
}


template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_precomp_rotary_embedding_cu(
	const int32 max_seq_len,
	const int32 emb_size,
	const int32 base = 10000)
{
	ACASSERT(emb_size % 2 == 0, "embedding size should be even");
	int32 d_per_2 = emb_size / 2;
	// initiate the frequence matrix
	Tensor<dtype, CUDA> freq(2, { max_seq_len, d_per_2 });
	dtype* freq_data = freq.buffer();

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_precomp_rotary_embedding_f32(max_seq_len, emb_size, base, freq);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return freq;
}


template<FloatingPointType dtype, Device device>
static Tensor<dtype, device> tensor_precomp_rotary_embedding(
	const int32 max_seq_len,
	const int32 emb_size,
	const int32 base = 10000)
{
	if constexpr (device == CPU)
	{
		return tensor_precomp_rotary_embedding_cpu<dtype>(max_seq_len, emb_size, base);
	}
	else
	{
		static_assert(device == CUDA, "unknown device");
		return tensor_precomp_rotary_embedding_cu<dtype>(max_seq_len, emb_size, base);
	}
}



/*
  Rotary position embedding operator.
  Applies the embedding on the input tensor.
  Article compatible version (https://arxiv.org/pdf/2104.09864)
  @param xt: input tensor (b, seq_len, [num_head], num_dims=2*emb_size)
  @param pt: positiion indices (b, seq_len)
  @param ft: angles for rotations (embedding tensor)
*/
template<PreciseFloatType dtype>
static Tensor<dtype, CPU> tensor_apply_rotary_embedding(
	const Tensor<dtype, CPU>& xt,
	const Tensor<int32, CPU>& pt,
	const Tensor<dtype, CPU>& ft)
{
	dtype* xt_data = xt.buffer();
	int32* pt_data = pt.buffer();
	dtype* ft_data = ft.buffer();

	// initiate the output
	Tensor<dtype, CPU> yt(xt.dim, xt.shape);
	dtype* yt_data = yt.buffer();

	ACASSERT(ft.dim == 2, "embedding tensor dim should be 2");
	int x_size = xt.size();
	int x_batch_stride = xt.stride[0];
	int x_seq_stride = xt.stride[1];
	int x_emb_stride = xt.stride[xt.dim - 2];  // stride corresponding to the embedding (last dim - 1)
	int x_emb_num = x_size / x_emb_stride;     // number of embeddings 
	int seq_len = xt.shape[1];
	
	int d_per_2 = ft.shape[1];
	int f_stride = ft.stride[0];

	int emb_offset = 0;
	for (int ix = 0; ix < x_emb_num; ++ix)
	{
		int b = emb_offset / x_batch_stride;
		int s = (emb_offset / x_seq_stride) % seq_len;  // sequence position, rotating index

		int m = pt_data[b * pt.stride[0] + s];

		for (int i = 0; i < d_per_2; ++i)
		{
			dtype angle = ft_data[m * f_stride + i];
			dtype cos_angle = cos(angle);
			dtype sin_angle = sin(angle);

			dtype x1 = xt_data[ix * x_emb_stride + i * 2];
			dtype x2 = xt_data[ix * x_emb_stride + i * 2 + 1];

			dtype y1 = cos_angle * x1 - sin_angle * x2;
			dtype y2 = sin_angle * x1 + cos_angle * x2;

			yt_data[ix * x_emb_stride + i * 2] = y1;
			yt_data[ix * x_emb_stride + i * 2 + 1] = y2;
		}

		emb_offset += x_emb_stride;
	}

	return yt;
}


template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_apply_rotary_embedding(
	const Tensor<dtype, CUDA>& xt,
	const Tensor<int32, CUDA>& pt,
	const Tensor<dtype, CUDA>& ft)
{
	// access the data arrays
	Tensor<dtype, CUDA> yt(xt.dim, xt.shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_apply_rotary_embedding_f32(xt, pt, ft, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}



/*
  Alternative Rotary position embedding operator.
  Applies the embedding on the input tensor.
  Alternative version that creates the pairs not from the
  consequtive elements but half dimension away.
  @param xt: input tensor (b, seq_len, [num_head], num_dims=2*emb_size)
  @param pt: positiion indices (b, seq_len)
  @param ft: angles for rotations (embedding tensor)
*/
template<PreciseFloatType dtype>
static Tensor<dtype, CPU> tensor_apply_alt_rotary_embedding(
	const Tensor<dtype, CPU>& xt,
	const Tensor<int32, CPU>& pt,
	const Tensor<dtype, CPU>& ft)
{
	dtype* xt_data = xt.buffer();
	int32* pt_data = pt.buffer();
	dtype* ft_data = ft.buffer();

	// initiate the output
	Tensor<dtype, CPU> yt(xt.dim, xt.shape);
	dtype* yt_data = yt.buffer();

	ACASSERT(ft.dim == 2, "embedding tensor dim should be 2");
	int x_size = xt.size();
	int x_batch_stride = xt.stride[0];
	int x_seq_stride = xt.stride[1];
	int x_emb_stride = xt.stride[xt.dim - 2];  // stride corresponding to the embedding (last dim - 1)
	int x_emb_num = x_size / x_emb_stride;     // number of embeddings 
	int seq_len = xt.shape[1];

	int d_per_2 = ft.shape[1];
	int f_stride = ft.stride[0];

	int emb_offset = 0;
	for (int ix = 0; ix < x_emb_num; ++ix)
	{
		int b = emb_offset / x_batch_stride;
		int s = (emb_offset / x_seq_stride) % seq_len;  // sequence position, rotating index

		int m = pt_data[b * pt.stride[0] + s];

		for (int i = 0; i < d_per_2; ++i)
		{
			dtype angle = ft_data[m * f_stride + i];
			dtype cos_angle = cos(angle);
			dtype sin_angle = sin(angle);

			dtype x1 = xt_data[ix * x_emb_stride + i];
			dtype x2 = xt_data[ix * x_emb_stride + d_per_2 + i];

			dtype y1 = cos_angle * x1 - sin_angle * x2;
			dtype y2 = sin_angle * x1 + cos_angle * x2;

			yt_data[ix * x_emb_stride + i] = y1;
			yt_data[ix * x_emb_stride + d_per_2 + i] = y2;
		}

		emb_offset += x_emb_stride;
	}

	return yt;
}


template<FloatingPointType dtype>  // TODO: implement
static Tensor<dtype, CUDA> tensor_apply_alt_rotary_embedding(
	const Tensor<dtype, CUDA>& xt,
	const Tensor<int32, CUDA>& pt,
	const Tensor<dtype, CUDA>& ft)
{
	// access the data arrays
	Tensor<dtype, CUDA> yt(xt.dim, xt.shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		//cu_tensor_embedding_f32(xt, wt, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}

#endif  // __EMBEDDING_OPS__
