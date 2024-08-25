#ifndef __CORE_CONCEPTS__
#define __CORE_CONCEPTS__

/*
   NVCC can not compile everything, e.g. concepts.
   Therefore this file is separated.
*/

#include "core.hpp"
#include <type_traits>
#include <concepts>

/*
  Concepts to group the data types.
*/

template<typename T>
concept FloatingPointType = std::floating_point<T> || std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>;

template<typename T>
concept ArithmeticType = (FloatingPointType<T> || std::integral<T>);

template<typename T>
concept IntegerType = std::integral<T>;

template<typename T>
concept HalfFloatType = (FloatingPointType<T> && sizeof(T) == 2);

template<typename T>
concept PreciseFloatType = (FloatingPointType<T> && sizeof(T) > 2);

template<typename T>
concept NotHalfFloatType = (std::integral<T> || (FloatingPointType<T> && sizeof(T) > 2));

#endif  // __CORE_CONCEPTS__