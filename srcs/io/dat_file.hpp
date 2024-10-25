#ifndef __DAT_FILE__
#define __DAT_FILE__

#include "tensor.hpp"

// DAT files are simple binary files, storing tensors
// 
// Tensor data file format(no compression) :
//    dimension - int(4 bytes)
//	  shape_1 - int(4 bytes)
//    ...
//    shape_dim - int(4 bytes)
//    data type - int(4 bytes)
//      . INT8 - 0
//      . INT16 - 1
//      . INT32 - 2
//      . BFLOAT16 - 3
//      . FLOAT16 - 4
//      . FLOAT32 - 5
//      . FLOAT64 - 6
//    tensor element 1 - data type(size depends on type)
//	  tensor element 2
//    ...
//    tensor element n(n can be calculated from shape)

// The file assumes default strides, no offset and default alignment.


/**
  Tensor reader from .dat files.
*/
template<typename T=float32>
static Tensor<T, CPU> load_tensor(const std::string& file_path)
{
	std::ifstream tensor_file(file_path, std::ios::binary);

	std::string msg = "Unable to open file: " + file_path;
	ACASSERT(tensor_file.is_open(), msg.c_str());

	// read the dimension
	int dim;
	tensor_file.read(reinterpret_cast<char*>(&dim), sizeof(int32));

	// read the shape
	std::vector<int64> shape(dim);
	for (int64 ix = 0; ix < dim; ++ix)
	{
		int axis_size;
		tensor_file.read(reinterpret_cast<char*>(&axis_size), sizeof(int32));
		shape[ix] = axis_size;
	}

	// read the dtype
	int dtype;
	tensor_file.read(reinterpret_cast<char*>(&dtype), sizeof(int32));

    if constexpr (std::is_same_v<T, float32>)
    {
        ACASSERT(dtype == 5, "Data type in tensor data file expected to be float32");
    }
    else
    {
		static_assert(std::is_same_v<T, int32>, "Unsupported data type");
        ACASSERT(dtype == 2, "Data type in tensor data file expected to be int32");
    }

	// read the data into a tensor
	Tensor<T, CPU> tensor(shape);

	int64 num_elements = calc_default_size(shape);
	size_t data_size = sizeof(T) * num_elements;
	size_t block_size = sizeof(T) * 2e6;

	//   read the data in chunks
	int64 offset = 0;
	while (offset < data_size)
	{
		size_t chunk_size = std::min(block_size, (size_t)((int64)data_size - offset));
		tensor_file.read(reinterpret_cast<char*>(tensor.buffer()) + offset, chunk_size);
		offset += chunk_size;
	}

	if (!tensor_file)
	{
		std::string msg = "Error during processing tensor data file: " + file_path;
		log_error(msg.c_str());
	}

	return tensor;
}


#endif  // __DAT_FILE__
