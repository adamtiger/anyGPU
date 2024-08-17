#include "datatypes.hpp"
#include <sstream>
#include <cassert>

std::string represent_datatype(const DataType dtype)
{
	static std::string names[] =
	{
		"INT8",
	    "INT16",
	    "INT32",
	    "BFLOAT16",
	    "FLOAT16",
	    "FLOAT32"
	};

	assert((int)dtype < 6);

	return names[(int)dtype];
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

	assert((int)device < 2);

	return names[(int)device];
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
