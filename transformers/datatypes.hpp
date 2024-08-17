#ifndef __DATA_TYPES__
#define __DATA_TYPES__

#include <type_traits>
#include <vector>
#include <array>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

const int MAX_TENSOR_DIM = 8;

using DimArray = std::array<int, MAX_TENSOR_DIM>;
using Shape = DimArray;
using Stride = DimArray;

enum Device
{
	CPU,
	CUDA
};

enum DataType
{
	INT8,
	INT16,
	INT32,
	BFLOAT16,
	FLOAT16,
	FLOAT32
};

static int bitsize_of_datatypes[] = {
	4, 8, 16, 32, 16, 16, 32
};

// int4 is 0.5 bytes long, from alignment perspective it is handled like 1
static int bytesize_of_datatypes[] = {
	1, 1, 2, 4, 2, 2, 4
};

using int8 = char;
using int16 = short signed int;
using int32 = signed int;
using bfloat16 = nv_bfloat16;
using float16 = half;
using float32 = float;

/*
  Returns the bitsize of the data type
*/
template<typename T> static int get_bitsize()
{
	return sizeof(T) * 8;
}

/*
  Returns the enum type of a data type.
  This function can help to reason about the
  underlying type of the tensor.
*/
template<typename T> static DataType get_datatype_enum()
{
	if constexpr (std::is_same_v<T, float32>)
	{
		return DataType::FLOAT32;
	}
	else if constexpr (std::is_same_v<T, float16>)
	{
		return DataType::FLOAT16;
	}
	else if constexpr (std::is_same_v<T, bfloat16>)
	{
		return DataType::BFLOAT16;
	}
	else if constexpr (std::is_same_v<T, int32>)
	{
		return DataType::INT32;
	}
	else if constexpr (std::is_same_v<T, int16>)
	{
		return DataType::INT16;
	}
	else if constexpr (std::is_same_v<T, int8>)
	{
		return DataType::INT8;
	}
	else
	{
		return DataType::UNK;
	}
}

/*
  Represents the data type enums as strings.
*/
std::string represent_datatype(const DataType dtype);

/*
  Represents the underlying type as string.
*/
template<typename dtype>
static std::string represent_datatype()
{
	DataType dt = get_datatype_enum<dtype>();
	return represent_datatype(dt);
}

/*
  Overriding the output stream op for DataType.
*/
std::ostream& operator<<(std::ostream& os, const DataType dtype);

/*
  Represents the device enums as strings.
*/
std::string represent_device(const Device device);

/*
  Overriding the output stream op for Device.
*/
std::ostream& operator<<(std::ostream& os, const Device device);

/**
  String representation of a shape or stride.
  It will be in the form of [d1, d2, d3] for 3 dimension.
  No new line character at the end.
*/
std::string represent_array(const int dim, const DimArray& arr);

/*
  Converting the int vector into any target data type.
*/
template<typename T>
static std::vector<T> cvt_int_vector_to_any(const std::vector<int>& hdata)
{
	std::vector<T> out_data(hdata.size());

	for (size_t ix = 0; ix < hdata.size(); ++ix)
	{
		out_data[ix] = static_cast<T>(hdata[ix]);
	}

	return out_data;
}

template<>
static std::vector<float16> cvt_int_vector_to_any(const std::vector<int>& hdata)
{
	std::vector<float16> out_data(hdata.size());

	for (size_t ix = 0; ix < hdata.size(); ++ix)
	{
		out_data[ix] = __int2half_rn(hdata[ix]);
	}

	return out_data;
}

template<>
static std::vector<bfloat16> cvt_int_vector_to_any(const std::vector<int>& hdata)
{
	std::vector<bfloat16> out_data(hdata.size());

	for (size_t ix = 0; ix < hdata.size(); ++ix)
	{
		out_data[ix] = __int2bfloat16_rn(hdata[ix]);
	}

	return out_data;
}

/*
  Converting an arbitrary type to float32.
  float32 is a precise enough format to represent
  any underlying type for printing.
*/
template<typename T>
static float32 cvt_any_to_float32(const T value)
{
	return static_cast<float32>(value);
}

template<>
static float32 cvt_any_to_float32<float16>(const float16 value)
{
	return __half2float(value);
}

template<>
static float32 cvt_any_to_float32<bfloat16>(const bfloat16 value)
{
	return __bfloat162float(value);
}


#endif  // __DATA_TYPES__
