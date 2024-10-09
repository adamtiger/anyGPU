#include "core.hpp"

bool equal(const int dim, const DimArray& lhs, const DimArray& rhs)
{
	bool eq = true;
	for (int idx = 0; idx < dim; ++idx)
	{
		eq = eq && (lhs[idx] == rhs[idx]);
	}
	return eq;
}

void increment_index(const int dim, const Shape& shape, Index& index)
{
	for (int dix = dim - 1; dix >= 0; --dix)
	{
		if (index[dix] >= shape[dix] - 1)
		{
			index[dix] = 0;
		}
		else
		{
			index[dix] += 1;
			break;
		}
	}
}

int calculate_offset(const int dim, const Stride& strides, const Index& index)
{
	int offset = 0;
	for (int ix = 0; ix < dim; ++ix)
	{
		offset += strides[ix] * index[ix];
	}
	return offset;
}

std::string represent_datatype(const DataType dtype)
{
	static std::string names[] =
	{
		"INT8",
	    "INT16",
	    "INT32",
	    "BFLOAT16",
	    "FLOAT16",
	    "FLOAT32",
		"FLOAT64",
		"FP8E4M3",
		"FP8E5M2"
	};

	ACASSERT((int)dtype < 9, "Unknown data type");

	return names[(unsigned int)dtype];
}

std::ostream& operator<<(std::ostream& os, const DataType dtype)
{
	os << represent_datatype(dtype);
	return os;
}

std::string represent_device(const Device device)
{
	static std::string names[] =
	{
		"CPU",
		"CUDA"
	};

	ACASSERT((int)device < 2, "Unknown device type");

	return names[(unsigned int)device];
}

std::ostream& operator<<(std::ostream& os, const Device device)
{
	os << represent_device(device);
	return os;
}

std::string represent_array(const int dim, const DimArray& arr)
{
	std::stringstream ss;

	ss << "[";
	for (int ix = 0; ix < dim - 1; ++ix)
	{
		ss << arr[ix] << ", ";
	}
	ss << arr[dim - 1];
	ss << "]";

	return ss.str();
}


unsigned int calc_req_num_blocks(unsigned int num_elements, unsigned int block_size)
{
	unsigned int temp = num_elements / block_size;
	temp += (num_elements % block_size > 0 ? 1 : 0);
	return temp;
}


std::string print_cuda_device_props()
{
	std::stringstream ss;
	ss << "--- CUDA device info --- \n";

	int num_devices = 0;
	cudaGetDeviceCount(&num_devices);

	cudaDeviceProp device_props;
	cudaGetDeviceProperties(&device_props, 0);

	ss << "  Device selection \n";
	ss << "    Num devices:                 " << num_devices << "\n";
	ss << "    Device id:                   " << 0 << "\n";  // for now, assume only one device
	ss << "    Device name:                 " << device_props.name << "\n";
	ss << "    Compute capability:          " << device_props.major << "." << device_props.minor << "\n";
	ss << "    integrated:                  " << device_props.integrated << "\n";
	ss << "\n";

	ss << "  Memory info \n";
	ss << "    sharedMemPerMultiprocessor:  " << device_props.sharedMemPerMultiprocessor << "\n";
	ss << "    sharedMemPerBlock:           " << device_props.sharedMemPerBlock << "\n";
	ss << "    memoryBusWidth:              " << device_props.memoryBusWidth << "\n";
	ss << "    totalGlobalMem:              " << device_props.totalGlobalMem << "\n";
	ss << "    totalConstMem:               " << device_props.totalConstMem << "\n";
	ss << "\n";

	ss << "  Kernel launch info \n";
	ss << "    maxGridSize.x:               " << device_props.maxGridSize[0] << "\n";
	ss << "    maxGridSize.y:               " << device_props.maxGridSize[1] << "\n";
	ss << "    maxGridSize.z:               " << device_props.maxGridSize[2] << "\n";
	ss << "    multiProcessorCount:         " << device_props.multiProcessorCount << "\n";
	ss << "    maxBlocksPerMultiProcessor:  " << device_props.maxBlocksPerMultiProcessor << "\n";
	ss << "    maxThreadsPerMultiProcessor: " << device_props.maxThreadsPerMultiProcessor << "\n";
	ss << "    maxThreadsPerBlock:          " << device_props.maxThreadsPerBlock << "\n";
	ss << "    regsPerMultiprocessor:       " << device_props.regsPerMultiprocessor << "\n";
	ss << "    regsPerBlock:                " << device_props.regsPerBlock << "\n";
	ss << "    concurrentKernels:           " << device_props.concurrentKernels << "\n";
	ss << "    asyncEngineCount:            " << device_props.asyncEngineCount << "\n";

	return ss.str();
}


int GlobalUUIDGenerator::next_id = 0;

int GlobalUUIDGenerator::generate_id()
{
	int temp = next_id;
	next_id += 1;
	return temp;
}


int calc_default_size(const std::vector<int>& shape)
{
	int tensor_size = 1;
	for (int s : shape)
	{
		tensor_size *= s;
	}
	return tensor_size;
}

int calc_default_size(const int dim, const DimArray& shape)
{
	int tensor_size = 1;
	for (int dix = 0; dix < dim; ++dix)
	{
		tensor_size *= shape[dix];
	}
	return tensor_size;
}

std::vector<int> calc_default_stride(const std::vector<int>& shape)
{
	int dim = calc_dim(shape);
	std::vector<int> stride(dim);
	stride[dim - 1] = 1;
	for (int dix = dim - 2; dix >= 0; --dix)
	{
		stride[dix] = stride[dix + 1] * shape[dix + 1];
	}
	return stride;
}

DimArray calc_default_stride(const int dim, const DimArray& shape)
{
	DimArray stride = {};
	stride[dim - 1] = 1;
	for (int dix = dim - 2; dix >= 0; --dix)
	{
		stride[dix] = stride[dix + 1] * shape[dix + 1];
	}
	return stride;
}

DimArray cvt_vector2array(const std::vector<int>& v)
{
	DimArray arr = {};
	int dim = calc_dim(v);
	for (int dix = 0; dix < dim; ++dix)
	{
		arr[dix] = v[dix];
	}
	return arr;
}

std::vector<int> cvt_array2vector(const int dim, const DimArray& arr)
{
	std::vector<int> v(dim);
	for (int dix = 0; dix < dim; ++dix)
	{
		v[dix] = arr[dix];
	}
	return v;
}

int calc_dim(const std::vector<int>& v)
{
	return static_cast<int>(v.size());
}
