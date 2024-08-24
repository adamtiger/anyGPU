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
concept ArithmeticType = (std::floating_point<T> || std::integral<T>);

template<typename T>
concept FloatingPointType = std::floating_point<T>;

template<typename T>
concept IntegerType = std::integral<T>;

template<typename T>
concept HalfFloatType = (std::floating_point<T> && sizeof(T) == 2);

template<typename T>
concept PreciseFloatType = (std::floating_point<T> && sizeof(T) > 2);

template<typename T>
concept NotHalfFloatType = (std::integral<T> || (std::floating_point<T> && sizeof(T) > 2));

#endif  // __CORE_CONCEPTS__