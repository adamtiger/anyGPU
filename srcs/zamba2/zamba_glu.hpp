#ifndef __ZAMBA_GLU__
#define __ZAMBA_GLU__

#include "zamba_glu.cuh"

#include "tensor.hpp"
#include "core_concepts.hpp"


/*
  GLU activation (elementwise op).
  Zamba2 compatible variant.
  Equivalent implementation:
      def glu(x):
          x = torch.chunk(x, 2, dim=-1)
          return F.gelu(x[0]) * x[1]
  Gelu impl:
      gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
*/
template<PreciseFloatType dtype>
static Tensor<dtype, CPU> tensor_zamba_glu(const Tensor<dtype, CPU>& xt)
{
	// access the data arrays
	int y_dim = xt.dim;
	Shape y_shape = xt.shape;
	y_shape[y_dim - 1] /= 2;  // end dimension will be half
	Tensor<dtype, CUDA> yt(y_dim, y_shape);
	dtype* y_data = yt.buffer();
	dtype* x_data = xt.buffer();

	// reference implementation
	// reliable (but slow)

	
	return yt;
}


template<FloatingPointType dtype>
static Tensor<dtype, CUDA> tensor_zamba_glu(const Tensor<dtype, CUDA>& xt)
{
	// access the data arrays
	int y_dim = xt.dim;
	Shape y_shape = xt.shape;
	y_shape[y_dim - 1] /= 2;  // end dimension will be half
	Tensor<dtype, CUDA> yt(y_dim, y_shape);

	if constexpr (std::is_same_v<dtype, float32>)
	{
		cu_tensor_zamba_glu_f32(xt, yt);
	}
	else
	{
		static_assert(std::is_same_v<dtype, float32>, "Unsupported data types");
	}

	return yt;
}


#endif  // __ZAMBA_GLU__
