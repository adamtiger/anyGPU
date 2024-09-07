#ifndef __TENSOR__
#define __TENSOR__

#include <memory>
#include <random>
#include "core.hpp"

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
	explicit MemoryBuffer(const int data_bit_size, const int dim, const Shape& shape);
	~MemoryBuffer();
};

template<Device device>
MemoryBuffer<device>::MemoryBuffer() : capacity(0)
{
	id = GlobalUUIDGenerator::generate_id();
	buffer = nullptr;
}

template<Device device>
MemoryBuffer<device>::MemoryBuffer(const int capacity) : capacity(capacity)
{
	id = GlobalUUIDGenerator::generate_id();
	buffer = nullptr;

	if constexpr (device == Device::CPU)
	{
		int n_align = (capacity >> ALIGNMENT_EXP);
		n_align += ((capacity & (ALIGNMENT_SIZE - 1)) == 0 ? 0 : 1);
		auto* temp = new AlignedType[n_align];
		buffer = reinterpret_cast<char*>(temp);
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
	id = GlobalUUIDGenerator::generate_id();

	int bitsize = calc_default_size(shape) * data_bit_size;
	capacity = (bitsize >> BYTE_EXP);
	capacity += ((bitsize & (BYTE_SIZE - 1)) == 0 ? 0 : 1);

	buffer = nullptr;

	if constexpr (device == Device::CPU)
	{
		int n_align = (capacity >> ALIGNMENT_EXP);
		n_align += ((capacity & (ALIGNMENT_SIZE - 1)) == 0 ? 0 : 1);
		auto* temp = new AlignedType[n_align];
		buffer = reinterpret_cast<char*>(temp);
		memset(buffer, 0, capacity);
	}
	else if constexpr (device == Device::CUDA)
	{
		cudaMalloc(&buffer, capacity);
		cudaMemset(buffer, 0, capacity);
	}
}

template<Device device>
MemoryBuffer<device>::MemoryBuffer(const int data_bit_size, const int dim, const Shape& shape)
{
	id = GlobalUUIDGenerator::generate_id();

	int bitsize = calc_default_size(dim, shape) * data_bit_size;
	capacity = (bitsize >> BYTE_EXP);
	capacity += ((bitsize & (BYTE_SIZE - 1)) == 0 ? 0 : 1);

	if constexpr (device == Device::CPU)
	{
		int n_align = (capacity >> ALIGNMENT_EXP);
		n_align += ((capacity & (ALIGNMENT_SIZE - 1)) == 0 ? 0 : 1);
		auto* temp = new AlignedType[n_align];
		buffer = reinterpret_cast<char*>(temp);
		memset(buffer, 0, capacity);
	}
	else if constexpr (device == Device::CUDA)
	{
		cudaMalloc(&buffer, capacity);
		cudaMemset(buffer, 0, capacity);
	}
}

template<Device device>
MemoryBuffer<device>::~MemoryBuffer()
{
	if (buffer == nullptr)
		return;

	if constexpr (device == Device::CPU)
	{
		delete[] reinterpret_cast<AlignedType*>(buffer);
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

	// returns the start of tensor elements
	//   takes into account the offset
	dtype* buffer() const
	{
		return reinterpret_cast<dtype*>(mem_buffer->buffer + offset);
	}

	// pointer of the memory buffer
	//   does not take into account the offset
	char* raw_buffer() const
	{
		return mem_buffer->buffer;
	}

	// number of elements in tensor
	//  no padding is included
	//  like the tensor stride is default
	int numel() const
	{
		return calc_default_size(dim, shape);
	}

	// number of elements in tensor
	//  includes the padding too (due to stride)
	int size() const
	{
		return shape[0] * stride[0];
	}

	// total size of the underlying memory (bytes)
	int capacity() const
	{
		return mem_buffer->capacity;
	}

	int buffer_id() const
	{
		return mem_buffer->id;
	}

	// deep copies the tensor to host
	Tensor<dtype, CPU> copy_to_host() const;

	// deep copies the tensor to cuda
	Tensor<dtype, CUDA> copy_to_cuda() const;

	explicit Tensor()
	{
		id = GlobalUUIDGenerator::generate_id();
		dim = 0;
		offset = 0;
		alignment = 1;
	}

	explicit Tensor(const int capacity)
	{
		id = GlobalUUIDGenerator::generate_id();
		dim = 0;
		offset = 0;
		alignment = 1;
		shape = {};
		stride = {};

		mem_buffer = std::make_shared<MemoryBuffer<device>>(capacity);
	}

	/*
	*  Creates a tensor with default strides and alignment (1 tensor element).
	*/
	explicit Tensor(const std::vector<int>& shape);
	
	/*
	*  Creates a tensor with default stride and alignment (1 tensor element).
	*/
	explicit Tensor(const int dim, const Shape& shape);

	/*
	*  Creates a tensor with default alignment (1 tensor element).
	*/
	explicit Tensor(const int dim, const Shape& shape, const Stride& stride);

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
	id = GlobalUUIDGenerator::generate_id();
	dim = calc_dim(shape);

	alignment = 1;
	offset = 0;

	mem_buffer = std::make_shared<MemoryBuffer<device>>(get_bitsize<dtype>(), shape);

	// calculating default stride
	auto stride = calc_default_stride(shape);
	this->shape = cvt_vector2array(shape);
	this->stride = cvt_vector2array(stride);
}

template<typename dtype, Device device>
Tensor<dtype, device>::Tensor(const int dim, const Shape& shape)
{
	id = GlobalUUIDGenerator::generate_id();
	this->dim = dim;

	alignment = 1;
	offset = 0;

	mem_buffer = std::make_shared<MemoryBuffer<device>>(get_bitsize<dtype>(), dim, shape);

	// calculating default stride
	this->stride = calc_default_stride(dim, shape);
	this->shape = shape;
}

template<typename dtype, Device device>
Tensor<dtype, device>::Tensor(const int dim, const Shape& shape, const Stride& stride)
{
	id = GlobalUUIDGenerator::generate_id();
	this->dim = dim;

	alignment = 1;
	offset = 0;

	int capacity = sizeof(dtype) * shape[0] * stride[0];
	mem_buffer = std::make_shared<MemoryBuffer<device>>(capacity);

	// calculating default stride
	this->shape = shape;
	this->stride = stride;
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

template<typename dtype, Device device>
Tensor<dtype, CPU> Tensor<dtype, device>::copy_to_host() const
{
	int capacity = this->mem_buffer->capacity;

	Tensor<dtype, CPU> tensor(capacity);
	tensor.dim = this->dim;
	tensor.shape = this->shape;
	tensor.stride = this->stride;
	tensor.alignment = this->alignment;
	tensor.offset = this->offset;
	
	char* trg = tensor.raw_buffer();
	char* src = this->raw_buffer();
	if constexpr (device == CPU)
	{
		memcpy(trg, src, capacity);
	}
	else if constexpr (device == CUDA)
	{
		cudaMemcpy(trg, src, capacity, cudaMemcpyDeviceToHost);
	}

	return tensor;
}

template<typename dtype, Device device>
Tensor<dtype, CUDA> Tensor<dtype, device>::copy_to_cuda() const
{
	int capacity = this->mem_buffer->capacity;

	Tensor<dtype, CUDA> tensor(capacity);
	tensor.dim = this->dim;
	tensor.shape = this->shape;
	tensor.stride = this->stride;
	tensor.alignment = this->alignment;
	tensor.offset = this->offset;

	char* trg = tensor.raw_buffer();
	char* src = this->raw_buffer();
	if constexpr (device == CPU)
	{
		cudaMemcpy(trg, src, capacity, cudaMemcpyHostToDevice);
	}
	else if constexpr (device == CUDA)
	{
		cudaMemcpy(trg, src, capacity, cudaMemcpyDeviceToDevice);
	}

	return tensor;
}

// tensor print, string representations

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
	ss << "  dtype:      " << get_datatype_enum<dtype>() << "\n";
	ss << "  dim:        " << tensor.dim << "\n";
	ss << "  alignment:  " << tensor.alignment << "\n";
	ss << "  offset:     " << tensor.offset << "\n";
	ss << "  shape:      " << represent_array(tensor.dim, tensor.shape) << "\n";
	ss << "  stride:     " << represent_array(tensor.dim, tensor.stride) << "\n";
	ss << "  capacity:   " << tensor.capacity() << "\n";

	ss << "  buffer content: \n";
	ss << "      [";

	int num_tensor_elements = tensor.size();
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

/**
  Creates a tensor filled with random numbers.
  The values are sampled from a uniform distribution
  from the range (0, 10).
*/
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

/**
  Creates a tensor filled with ones.
*/
template<typename dtype, Device device>
static Tensor<dtype, device> crt_ones_tensor(const std::vector<int>& shape)
{
	int length = calc_default_size(shape);
	std::vector<int> host_data(length);
	for (int ix = 0; ix < length; ++ix)
	{
		host_data[ix] = 1;
	}

	auto dtype_hdata = cvt_int_vector_to_any<dtype>(host_data);

	Tensor<dtype, device> tensor(shape, dtype_hdata);
	return tensor;
}


/**
  Tensor reader from .dat files.
*/
static Tensor<float32, CPU> load_tensor(const std::string& file_path)
{
	std::ifstream tensor_file(file_path, std::ios::binary);  // TODO: check file open success

	// read the dimension
	int dim;
	tensor_file.read(reinterpret_cast<char*>(&dim), sizeof(int32));

	// read the shape
	std::vector<int> shape(dim);
	for (int ix = 0; ix < dim; ++ix)
	{
		int axis_size;
		tensor_file.read(reinterpret_cast<char*>(&axis_size), sizeof(int32));
		shape[ix] = axis_size;
	}

	// read the dtype (it has to be float32)
	int dtype;
	tensor_file.read(reinterpret_cast<char*>(&dtype), sizeof(int32));
	assert(dtype == 5);
	
	// read the data
	int num_elements = calc_default_size(shape);
	std::vector<float32> tensor_data(num_elements);
	tensor_file.read(reinterpret_cast<char*>(tensor_data.data()), sizeof(float32) * num_elements);

	if (tensor_file)
	{
		// TODO: add logger, and provide customizable feedback
		//std::cout << "Tensor data was read successfully." << std::endl;
	}

	Tensor<float32, CPU> tensor(shape, tensor_data);
	return tensor;
}


/**
  Comparators for tensors.
*/

/**
  Decides if two tensors have the same ordering of elements 
  in memory buffer.
  This is important for elementwise operations because
  the element offsets will be the same in both tensors.
*/
template<typename dtype, Device device>
static bool elementwise_compatible(const Tensor<dtype, device>& lhs, const Tensor<dtype, device>& rhs)
{
	bool cp = true;
	cp = cp && (lhs.dim == rhs.dim);
	cp = cp && equal(lhs.dim, lhs.shape, rhs.shape);
	cp = cp && equal(lhs.dim, lhs.stride, rhs.stride);
	return cp;
}

/**
  Decides if two tensors can be matrix multiplied by each other.
*/
template<typename dtype, Device device>
static bool matmul_compatible(const Tensor<dtype, device>& lhs, const Tensor<dtype, device>& rhs)
{
	bool cp = true;
	cp = cp && (lhs.dim == 2);
	cp = cp && (rhs.dim == 2);
	cp = cp && (lhs.shape[1] == rhs.shape[0]);
	return cp;
}

/**
  Automatically calculates the grid and block sizes for
  elementwise (pointwise) operations.
*/
template<typename dtype, Device device>
static KernelParameters calc_kernel_prms_pointwise(const Tensor<dtype, device>& tensor)
{
	int num_operations = tensor.size();
	unsigned int threads_per_block = 32 * 8;

	unsigned int num_grids = num_operations / threads_per_block;
	num_grids += ((num_operations % threads_per_block > 0) ? 1 : 0);

	KernelParameters kpms = {};
	kpms.block_size = { threads_per_block, 1, 1 };
	kpms.grid_size = { num_grids, 1, 1 };

	return kpms;
}

#endif  // __TENSOR__
