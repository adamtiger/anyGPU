#ifndef __TENSOR__
#define __TENSOR__

#include <vector>
#include <array>

#include <cuda_runtime.h>
#include "utils.hpp"

#include <iostream>

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

// helper functions for tensor size calculations

/*
*  If the alignment is the default for the given data type,
*  this function returns the number of elements in the tensor.
*  Default alignment means the byte size of 1 tensor element.
*/
int calculate_default_size(const std::vector<int>& shape);


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

	int bitsize = calculate_default_size(shape) * data_bit_size;
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
*  mem_buffer - a MemoryBuffer object
*/
template<DataType dtype, Device device=Device::CUDA>
struct Tensor
{
	int id;
	int dim;
	int offset;
	int alignment;  // in bytes
	Shape shape;
	Stride stride;

	MemoryBuffer<device> mem_buffer;

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
};

template<DataType dtype, Device device>
Tensor<dtype, device>::Tensor(const std::vector<int>& shape) : mem_buffer(bitsize_of_datatypes[(int)dtype], shape)
{
	id = GlobalUUDGenerator::generate_id();
	dim = static_cast<int>(shape.size());

	alignment = bytesize_of_datatypes[(int)device];
	offset = 0;

	// calculating default stride
	this->shape[dim - 1] = shape[dim - 1];
	this->stride[dim - 1] = 1;
	for (int dix = dim - 2; dix >= 0; --dix)
	{
		this->shape[dix] = shape[dix];
		this->stride[dix] = this->stride[dix + 1] * shape[dix];
	}
}

// random tensor creator

// tensor creator with a pattern



#endif  // __TENSOR__
