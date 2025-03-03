#ifndef __ATTENTION_TOOLS__
#define __ATTENTION_TOOLS__

#include "tensor.hpp"
#include "core_concepts.hpp"

template<FloatingPointType T>
inline T calculate_alpha(const int d)
{
	T alpha = static_cast<T>(1.f / sqrtf(static_cast<float32>(d)));
	return alpha;
}

template<>
inline float16 calculate_alpha(const int d)
{
	float16 alpha = __float2half(1.f / sqrtf(static_cast<float32>(d)));
	return alpha;
}

template<>
inline bfloat16 calculate_alpha(const int d)
{
	bfloat16 alpha = __float2bfloat16(1.f / sqrtf(static_cast<float32>(d)));
	return alpha;
}

template<>
inline fp8e4m3 calculate_alpha(const int d)
{
	fp8e4m3 alpha(1.f / sqrtf(static_cast<float32>(d)));
	return alpha;
}

template<>
inline fp8e5m2 calculate_alpha(const int d)
{
	fp8e5m2 alpha(1.f / sqrtf(static_cast<float32>(d)));
	return alpha;
}

#endif  // __ATTENTION_TOOLS__
