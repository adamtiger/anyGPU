#ifndef __SAFE_TENSORS_FILE__
#define __SAFE_TENSORS_FILE__

#include "tensor.hpp"

struct TensorInfo
{
	unsigned long long offset_start;
	unsigned long long offset_end;
	int dim;
	Shape shape;
	DataType dtype;
	std::string name;
};

std::ostream& operator<<(std::ostream& os, const TensorInfo& tensor_info);

/*
  Reads the tensor infos from the json string.
*/
void _sft_read_tensorinfo_from_json(
	const std::string& header_content, 
	std::vector<TensorInfo>& tensor_infos);


/* 
  Reads the tensors from the safetensors file.
*/
template<typename T>
static void sft_read_tensors(const std::string& path, std::vector<Tensor<T, CPU>>& tensors)
{
	std::ifstream sft_file(path, std::ios::binary);
	ACASSERT(sft_file.is_open(), "safetensors file was not opened!");

	// read tensor infos
	std::vector<TensorInfo> tensor_infos;
	{
		// read the size of the header (bytes)
		unsigned long long header_size;
		sft_file.read(reinterpret_cast<char*>(&header_size), sizeof(unsigned long long));

		// read the header content
		std::vector<char> header_raw_content(header_size);
		sft_file.read(header_raw_content.data(), header_size);
		std::string header_string(header_raw_content.data());

		std::cout << header_string << std::endl;

		// read the tensor infos from json string
		_sft_read_tensorinfo_from_json(
			header_string,
			tensor_infos
		);
	}

	std::cout << tensor_infos[0];
	std::cout << tensor_infos[1];
	std::cout << tensor_infos[2];

	// create tensors from tensor infos
	size_t prev_offset_end = 0;
	tensors.reserve(tensor_infos.size());
	for (auto tensor_info : tensor_infos)
	{
		ACASSERT(tensor_info.dtype == get_datatype_enum<T>(), "unexpected data type");

		// init tensor
		Tensor<T, CPU> tensor(tensor_info.dim, tensor_info.shape);
		tensor.name = tensor_info.name;

		// move the position in the file to the right offset
		sft_file.seekg(tensor_info.offset_start - prev_offset_end, sft_file.cur);

		// copy data into tensor from file
		size_t buffer_size = tensor_info.offset_end - tensor_info.offset_start;
		sft_file.read(reinterpret_cast<char*>(tensor.buffer()), buffer_size);

		// save tensor in vector
		tensors.push_back(tensor);
	}
}

#endif  // __SAFE_TENSORS_FILE__
