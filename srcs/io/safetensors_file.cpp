#include "safetensors_file.hpp"

using u64 = unsigned long long;

constexpr char LEFT_CURL = '{';
constexpr char RIGHT_CURL = '}';
constexpr char LEFT_RECT = '[';
constexpr char RIGHT_RECT = ']';
constexpr char COLON = ':';
constexpr char HYPEN = '"';
constexpr char COMMA = ',';

std::string extract_expected_name(const std::string& header_content, size_t& cindex)
{
	size_t first = header_content.find_first_of(HYPEN, cindex);
	size_t second = header_content.find_first_of(HYPEN, first+1);

	cindex = second + 1;

	return header_content.substr(first + 1, second - first - 1);
}

DataType extract_data_type(const std::string& header_content, size_t& cindex)
{
	static std::unordered_map<std::string, DataType> sf_dtype_map =
	{
		{"F64", DataType::FLOAT64},
		{"F32", DataType::FLOAT32},
		{"F16", DataType::FLOAT16},
		{"BF16", DataType::BFLOAT16},
		{"I32", DataType::INT32},
		{"I16", DataType::INT16},
		{"I8", DataType::INT8}
	};

	auto sf_dtype_name = extract_expected_name(header_content, cindex);

	ACASSERT(sf_dtype_map.contains(sf_dtype_name), "unsupported data type in safetensors");

	return sf_dtype_map.at(sf_dtype_name);
}

std::vector<int> extract_list(const std::string& header_content, size_t& cindex)
{
	size_t first = header_content.find_first_of(LEFT_RECT, cindex);
	size_t second = header_content.find_first_of(RIGHT_RECT, first + 1);

	cindex = second + 1;

	std::string snum = "";
	std::vector<int> shape;
	int cix = first + 1;

	while (cix <= second)
	{
		char c = header_content[cix];
		if (std::isdigit(c))
		{
			snum.push_back(c);
		}
		else
		{
			if (!snum.empty())
			{
				shape.push_back(std::atoi(snum.c_str()));
				snum.clear();
			}
		}

		cix++;
	}

	return shape;
}


void _sft_read_tensorinfo_from_json(
	const std::string& header_content,
	std::vector<TensorInfo>& tensor_infos)
{
	size_t cindex = 0;

	// consume first LEFT_CURL (beginning of json)
	cindex = header_content.find_first_of(LEFT_CURL, cindex);

	while (cindex != std::string::npos)
	{
		// find next name or (__metadata__)
		auto name = extract_expected_name(header_content, cindex);
		if (name != "__metadata__")  // process the tensor info
		{
			TensorInfo tensor_info = {};
			tensor_info.name = name;

			for (int i = 0; i < 3; ++i)
			{
				auto key = extract_expected_name(header_content, cindex);

				if (key == "dtype")
				{
					tensor_info.dtype = extract_data_type(header_content, cindex);
				}
				else if (key == "shape")
				{
					auto shape = extract_list(header_content, cindex);
					tensor_info.dim = static_cast<int>(shape.size());
					tensor_info.shape = cvt_vector2array(shape);
				}
				else if (key == "data_offsets")
				{
					auto offsets = extract_list(header_content, cindex);
					ACASSERT(offsets.size() == 2, "exactly two offset value is needed");
					tensor_info.offset_start = offsets[0];
					tensor_info.offset_end = offsets[1];
				}
			}

			tensor_infos.push_back(tensor_info);
		}
		else  // skip the metadata
		{
			cindex = header_content.find_first_of(RIGHT_CURL, cindex);
		}

		// after each tensor info there is a comma, except the last one
		cindex = header_content.find_first_of(COMMA, cindex);  
	}
}

std::ostream& operator<<(std::ostream& os, const TensorInfo& tensor_info)
{
	os << "Tensor name: " << tensor_info.name << "\n";
	os << "  shape:     " << represent_array(tensor_info.dim, tensor_info.shape) << "\n";
	os << "  dtype:     " << represent_datatype(tensor_info.dtype) << "\n";
	os << "  start:     " << tensor_info.offset_start << "\n";
	os << "  end:       " << tensor_info.offset_end << "\n";
	return os;
}

