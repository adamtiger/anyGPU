#ifndef __TENSOR__
#define __TENSOR__

#include <sstream>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <random>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "utils.hpp"


const int MAX_TENSOR_DIM = 8;

using Shape = std::array<int, MAX_TENSOR_DIM>;
using Stride = std::array<int, MAX_TENSOR_DIM>;

enum Device
{
	CPU,
	CUDA
};

enum DataType
{
	INT4,
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

// helper functions for tensor size calculations

/*
*  If the alignment is the default for the given data type,
*  this function returns the number of elements in the tensor.
*  Default alignment means the byte size of 1 tensor element.
*/
int calc_default_size(const std::vector<int>& shape);

/*
*  Calculates the default stride from the given shape.
*/
std::vector<int> calc_default_stride(const std::vector<int>& shape);

/*
*  Transforms int vector to array.
*  Vector can be shape or stride.
*/
std::array<int, MAX_TENSOR_DIM> cvt_vector2array(const std::vector<int>& v);

/*
*  Dimension from vector.
*/
int calc_dim(const std::vector<int>& v);

/*
*  Returns the bitsize of the data type
*/
template<typename T> static int get_bitsize()
{
	return sizeof(T) * 8;
}

/*
*  Converting the int vector into any target data type.
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

/*
*  Contains a flat memory for storing the tensor data.
*  Can live in more than one tensors.
*  Template: device - on which device the memory was allocated
*    This ensures the memory on different devices are handled
*    separately, as they are different obect types.
*  
*  id - unique identifier for memory (two buffer can not overlap with different ids)
*  capacity - the size of the memory in bytes
*  buffer - address of memory (always aligned to 256 bytes)
*/
template<Device device>
struct MemoryBuffer
{
	int id;
	int capacity;
	char* buffer;

	explicit MemoryBuffer();
	explicit MemoryBuffer(const int capacity);
	explicit MemoryBuffer(const int data_bit_size, const std::vector<int>& shape);
	~MemoryBuffer();
};

template<Device device>
MemoryBuffer<device>::MemoryBuffer() : capacity(0)
{
	id = GlobalUUDGenerator::generate_id();
	buffer = nullptr;
}

template<Device device>
MemoryBuffer<device>::MemoryBuffer(const int capacity) : capacity(capacity)
{
	id = GlobalUUDGenerator::generate_id();

	if constexpr (device == Device::CPU)
	{
		buffer = new char[capacity];
		memset(buffer, 0, capacity);
	}
	else if constexpr (device == Device::CUDA)
	{
		cudaMalloc(&buffer, capacity);
		cudaMemset(buffer, 0, capacity);
	}
}

template<Device device>
MemoryBuffer<device>::MemoryBuffer(const int data_bit_size, const std::vector<int>& shape)
{
	id = GlobalUUDGenerator::generate_id();

	int bitsize = calc_default_size(shape) * data_bit_size;
	capacity = (bitsize >> 3);
	capacity += ((bitsize & 0x00000007) == 0 ? 0 : 1);

	if constexpr (device == Device::CPU)
	{
		buffer = new char[capacity];
	}
	else if constexpr (device == Device::CUDA)
	{
		cudaMalloc(&buffer, capacity);
	}
}

template<Device device>
MemoryBuffer<device>::~MemoryBuffer()
{
	if (buffer == nullptr)
		return;

	if constexpr (device == Device::CPU)
	{
		delete[] buffer;
	}
	else if constexpr (device == Device::CUDA)
	{
		cudaFree(buffer);
	}
	
	buffer = nullptr;
	capacity = 0;
}


/*
*  Tensors are views of a flat memory buffer.
*  Handling tensors separately from the underlying
*  buffer provides a natural way to handle changes
*  in the views (e.g. slicing without copy).
*  Several tensor can have the same underlying buffer.
*  
*  id - unique identifier of the tensor
*  dim - dimension of the tensor
*  offset - offset inside the memory buffer (start position of the tensor)
*  alignment - alignment requirement of the tensor internal axis (padding may be applied)
*  shape - shape of the tensor
*  stride - stride of the tensor
*  data - pointer to the buffer memory
*  mem_buffer - a MemoryBuffer object (for tracking the memory)
*/
template<typename dtype, Device device=Device::CUDA>
struct Tensor
{
	int id;
	int dim;
	int offset;     // in bytes
	int alignment;  // in bytes
	Shape shape;
	Stride stride;

	dtype* buffer() const
	{
		return reinterpret_cast<dtype*>(mem_buffer->buffer + offset);
	}

	int numel() const
	{
		return shape[0] * stride[0];
	}

	explicit Tensor()
	{
		id = GlobalUUDGenerator::generate_id();
		dim = 0;
		offset = 0;
		alignment = 1;
	}

	/*
	*  Creates a tensor with default strides and alignment (1 tensor element).
	*/
	explicit Tensor(const std::vector<int>& shape);

	/*
	*  Creates a tensor with default strides and alignment.
	*  Also uses the tensor values given on the host.
	*/
	explicit Tensor(const std::vector<int>& shape, const std::vector<dtype>& hdata);

private:
	std::shared_ptr<MemoryBuffer<device>> mem_buffer;
};

template<typename dtype, Device device>
Tensor<dtype, device>::Tensor(const std::vector<int>& shape)
{
	id = GlobalUUDGenerator::generate_id();
	dim = calc_dim(shape);

	alignment = bytesize_of_datatypes[(int)device];
	offset = 0;

	mem_buffer = std::make_shared<MemoryBuffer<device>>(get_bitsize<dtype>(), shape);

	// calculating default stride
	auto stride = calc_default_stride(shape);
	this->shape = cvt_vector2array(shape);
	this->stride = cvt_vector2array(stride);
}

template<typename dtype, Device device>
Tensor<dtype, device>::Tensor(const std::vector<int>& shape, const std::vector<dtype>& hdata) : Tensor(shape)
{
	// copy the data
	size_t data_size = sizeof(dtype) * hdata.size();
	if constexpr (device == Device::CPU)
	{
		memcpy(mem_buffer->buffer, hdata.data(), data_size);
	}
	else if  constexpr (device == Device::CUDA)
	{
		cudaMemcpy(mem_buffer->buffer, hdata.data(), data_size, cudaMemcpyHostToDevice);
	}
}

// tensor print, string representations

/**
  String representation of a shape or stride.
  It will be in the form of [d1, d2, d3] for 3 dimension.
  No new line character at the end.
*/
std::string represent_array(const int dim, const std::array<int, MAX_TENSOR_DIM>& arr);

/**
  String representation of tensor.
  Multiline format, in-depth information.
  @param tensor: only CPU tensors can be printed
  @param head: the number of elements to print from the buffer (from the head of the buffer)
*/
template<typename dtype>
static std::string represent_tensor(const Tensor<dtype, CPU>& tensor, const int head=5)
{
	std::stringstream ss;

	ss << "-- Tensor -- \n";
	ss << "  id:         " << tensor.id << "\n";
	ss << "  dim:        " << tensor.dim << "\n";
	ss << "  alignment:  " << tensor.alignment << "\n";
	ss << "  offset:     " << tensor.offset << "\n";
	ss << "  shape:      " << represent_array(tensor.dim, tensor.shape) << "\n";
	ss << "  stride:     " << represent_array(tensor.dim, tensor.stride) << "\n";

	ss << "  buffer content: \n";
	ss << "      [";

	int num_tensor_elements = tensor.numel();
	int num_data_to_print = std::min(head, num_tensor_elements);

	dtype* data = tensor.buffer();
	for (int ix = 0; ix < num_data_to_print - 1; ++ix)
	{
		ss << cvt_any_to_float32(data[ix]) << ", ";
	}
	ss << cvt_any_to_float32(data[num_data_to_print - 1]);

	if (num_data_to_print < num_tensor_elements)
	{
		ss << ", ...";  // show not all elements were printed
	}

	ss << "] \n";

	return ss.str();
}

// random tensor creator
template<typename dtype, Device device>
static Tensor<dtype, device> crt_random_tensor(const std::vector<int>& shape, const int seed=10)
{
	std::default_random_engine eng(seed);
	std::uniform_int_distribution<int> uni_dist(0, 10);

	int length = calc_default_size(shape);
	std::vector<int> host_data(length);
	for (int ix = 0; ix < length; ++ix)
	{
		host_data[ix] = uni_dist(eng);
	}

	auto dtype_hdata = cvt_int_vector_to_any<dtype>(host_data);

	Tensor<dtype, device> tensor(shape, dtype_hdata);

	return tensor;
}

/** 
  Tensor creator with a fixed pattern.
  Good for testing the result in other 
  frameworks like python, because the 
  tensors can be easily reproduced.
*/
template<typename dtype, Device device>
static Tensor<dtype, device> crt_pattern_tensor(const std::vector<int>& shape)
{
	int length = calc_default_size(shape);
	std::vector<int> host_data(length);
	for (int ix = 0; ix < length; ++ix)
	{
		host_data[ix] = ix % 10;
	}

	auto dtype_hdata = cvt_int_vector_to_any<dtype>(host_data);

	Tensor<dtype, device> tensor(shape, dtype_hdata);

	return tensor;
}

#endif  // __TENSOR__
