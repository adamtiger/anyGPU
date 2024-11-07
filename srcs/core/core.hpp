#ifndef __CORE__
#define __CORE__

#include <unordered_map>
#include <vector>
#include <array>
#include <functional>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <cmath>
#include <regex>

#ifdef __INTELLISENSE__  // to have intellisense help for wmma, threadIdx etc.
#define __CUDACC__
#define __CUDA_ARCH__ 860
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>  // for tensor cores


/* data type related */

using int8 = char;
using int16 = short signed int;
using int32 = signed int;
using int64 = long long;
using bfloat16 = nv_bfloat16;
using float16 = half;
using float32 = float;
using float64 = double;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

static long long bitsize_of_datatypes[] = {
	8, 16, 32, 64, 16, 16, 32, 64, 8, 8
};

// int4 is 0.5 bytes long, from alignment perspective it is handled like 1
static long long bytesize_of_datatypes[] = {
	1, 2, 4, 8, 2, 2, 4, 8, 1, 1
};

/*
  Returns the bitsize of the data type
*/
template<typename T> inline int64 get_bitsize()
{
	return static_cast<int64>(sizeof(T) * 8);
}

/* Shape related */

const int MAX_TENSOR_DIM = 8;

using DimArray = std::array<int64, MAX_TENSOR_DIM>;
using Shape = DimArray;
using Stride = DimArray;
using Index = DimArray;

bool equal(const int64 dim, const DimArray& lhs, const DimArray& rhs);
void increment_index(const int64 dim, const Shape& shape, Index& index);
int64 calculate_offset(const int64 dim, const Stride& strides, const Index& index);

/* Enums */

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
    INT64,
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    FP8E4M3,
    FP8E5M2
};


/*
  Returns the enum type of a data type.
  This function can help to reason about the
  underlying type of the tensor.
*/
template<typename T> inline DataType get_datatype_enum()
{
	if constexpr (std::is_same_v<T, float32>)
	{
		return DataType::FLOAT32;
	}
	else if constexpr (std::is_same_v<T, float64>)
	{
		return DataType::FLOAT64;
	}
	else if constexpr (std::is_same_v<T, float16>)
	{
		return DataType::FLOAT16;
	}
	else if constexpr (std::is_same_v<T, bfloat16>)
	{
		return DataType::BFLOAT16;
	}
	else if constexpr (std::is_same_v<T, fp8e4m3>)
	{
		return DataType::FP8E4M3;
	}
	else if constexpr (std::is_same_v<T, fp8e5m2>)
	{
		return DataType::FP8E5M2;
	}
	else if constexpr (std::is_same_v<T, int32>)
	{
		return DataType::INT32;
	}
	else if constexpr (std::is_same_v<T, int64>)
	{
		return DataType::INT64;
	}
	else if constexpr (std::is_same_v<T, int16>)
	{
		return DataType::INT16;
	}
	else
	{
		static_assert(std::is_same_v<T, int8>, "unknown data type");
		return DataType::INT8;
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
inline std::string represent_datatype()
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
std::string represent_array(const int64 dim, const DimArray& arr);

/*
  Converting the int vector into any target data type.
*/
template<typename T>
inline std::vector<T> cvt_int_vector_to_any(const std::vector<int>& hdata)
{
	std::vector<T> out_data(hdata.size());

	for (size_t ix = 0; ix < hdata.size(); ++ix)
	{
		out_data[ix] = static_cast<T>(hdata[ix]);
	}

	return out_data;
}

//template<>
//static std::vector<float16> cvt_int_vector_to_any(const std::vector<int>& hdata)
//{
//	std::vector<float16> out_data(hdata.size());
//
//	for (size_t ix = 0; ix < hdata.size(); ++ix)
//	{
//		out_data[ix] = __int2half_rn(hdata[ix]);
//	}
//
//	return out_data;
//}

template<>
inline std::vector<bfloat16> cvt_int_vector_to_any(const std::vector<int>& hdata)
{
	std::vector<bfloat16> out_data(hdata.size());

	for (size_t ix = 0; ix < hdata.size(); ++ix)
	{
		out_data[ix] = __int2bfloat16_rn(hdata[ix]);
	}

	return out_data;
}

template<>
inline std::vector<fp8e4m3> cvt_int_vector_to_any(const std::vector<int>& hdata)
{
	std::vector<fp8e4m3> out_data(hdata.size());

	for (size_t ix = 0; ix < hdata.size(); ++ix)
	{
		fp8e4m3 temp(hdata[ix]);
		out_data[ix] = temp;
	}

	return out_data;
}

template<>
inline std::vector<fp8e5m2> cvt_int_vector_to_any(const std::vector<int>& hdata)
{
	std::vector<fp8e5m2> out_data(hdata.size());

	for (size_t ix = 0; ix < hdata.size(); ++ix)
	{
		fp8e5m2 temp(hdata[ix]);
		out_data[ix] = temp;
	}

	return out_data;
}

/*
  Converting an arbitrary type to float32.
  float32 is a precise enough format to represent
  any underlying type for printing.
*/
template<typename T>
inline float32 cvt_any_to_float32(const T value)
{
	return static_cast<float32>(value);
}

template<>
inline float32 cvt_any_to_float32<float16>(const float16 value)
{
	return __half2float(value);
}

template<>
inline float32 cvt_any_to_float32<bfloat16>(const bfloat16 value)
{
	return __bfloat162float(value);
}

template<>
inline float32 cvt_any_to_float32<fp8e4m3>(const fp8e4m3 value)
{
	return (float)value;
}

template<>
inline float32 cvt_any_to_float32<fp8e5m2>(const fp8e5m2 value)
{
	return (float)value;
}


/* Miscallenous */

/*
  Generates global universal unique ids.
*/
class GlobalUUIDGenerator
{
public:
	static int64 generate_id();

private:
	static int64 next_id;
};


constexpr int64 BYTE_EXP = 3;
constexpr int64 BYTE_SIZE = 8;

constexpr int64 ALIGNMENT_EXP = 8;
constexpr int64 ALIGNMENT_SIZE = 256;

/*
  Data type for 256 bytes aligned allocation.
*/
struct alignas(ALIGNMENT_SIZE) AlignedType
{
	char data[ALIGNMENT_SIZE];
};


// helper functions for tensor size calculations

/*
  If the alignment is the default for the given data type,
  this function returns the number of elements in the tensor.
  Default alignment means the byte size of 1 tensor element.
*/
int64 calc_default_size(const std::vector<int64>& shape);

/*
  If the alignment is the default for the given data type,
  this function returns the number of elements in the tensor.
  Default alignment means the byte size of 1 tensor element.
*/
int64 calc_default_size(const int64 dim, const DimArray& shape);

/*
  Calculates the default stride from the given shape.
*/
std::vector<int64> calc_default_stride(const std::vector<int64>& shape);

/*
  Calculates the default stride from the given shape.
*/
DimArray calc_default_stride(const int64 dim, const DimArray& shape);

/*
  Transforms int vector to array.
  Vector can be shape or stride.
*/
DimArray cvt_vector2array(const std::vector<int64>& v);

/*
  Transforms int array to vector.
  Vector can be shape or stride.
*/
std::vector<int64> cvt_array2vector(const int64 dim, const DimArray& arr);

/*
  Dimension from vector.
*/
int64 calc_dim(const std::vector<int64>& v);


/* logger, assert */

enum class LogLevel
{
	LEVEL_INFO,
	LEVEL_WARNING,
	LEVEL_ERROR
};

static void log_message(LogLevel llevel, const std::string& msg)
{
	// static handlers
	static const std::string log_prefix_table[] =
	{
		"\033[33m[INFO    ",
		"\033[36m[WARNING ",
		"\033[91m[ERROR   "
	};

	std::stringstream ss;
	ss << log_prefix_table[int(llevel)] << "] ";
	ss << msg;

	// write to screen
	std::cout << ss.str() << "\033[m" << std::endl;
}

static void log_info(const char* cmsg)
{
	std::string msg(cmsg);
	log_message(LogLevel::LEVEL_INFO, msg);
}

static void log_warning(const char* cmsg)
{
	std::string msg(cmsg);
	log_message(LogLevel::LEVEL_WARNING, msg);
}

static void log_error(const char* cmsg)
{
	std::string msg(cmsg);
	log_message(LogLevel::LEVEL_ERROR, msg);
}

#ifdef AC_WITH_ASSERT
#define ACASSERT( expression, msg )         \
    do {                                    \
        bool cond = (expression);           \
	    if (cond) {}                \
		else                                \
		{                                   \
			std::cout << __FILE__ << " " << __LINE__ << '\n';  \
			log_error(msg);                 \
		    exit(1);                        \
		}                                   \
	} while(0)
#else
#define ACASSERT( expression, msg ) ((void)0)
#endif


/* Cuda related */

// The size of the grid and block.
struct KernelParameters
{
	dim3 grid_size;
	dim3 block_size;
};

// Calculates the number of required blocks for a specific number of elements
unsigned int calc_req_num_blocks(unsigned int num_elements, unsigned int block_size);

/*
  Prints the device properties of the cuda device.
*/
std::string print_cuda_device_props();

/*
  Cuda error checks.
*/
#define CUDA_CHECK( fn )                      \
    do {                                      \
        cudaError_t res = (fn);               \
	    if (res == cudaSuccess) { }           \
		else                                  \
		{                                     \
			std::stringstream ss;             \
            ss << "code: " << (int)res;       \
            ss << " file: " << __FILE__;      \
            ss << " ln: " << __LINE__;        \
            ss << "\n";                       \
            log_error(ss.str().c_str());      \
            exit(1);                          \
		}                                     \
	} while (0)

#define CUDA_CHECK_LAST_ERROR( )              \
    do {                                      \
        cudaError_t res = cudaGetLastError(); \
	    if (res == cudaSuccess) { }           \
		else                                  \
		{                                     \
			std::stringstream ss;             \
            ss << "code: " << (int)res;       \
            ss << " file: " << __FILE__;      \
            ss << " ln: " << __LINE__;        \
            ss << "\n";                       \
            log_error(ss.str().c_str());      \
            exit(1);                          \
		}                                     \
	} while (0)

#endif  // __CORE__
