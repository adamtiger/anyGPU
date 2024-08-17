#include <vector>
#include <random>
#include <iostream>

#include "tensor.hpp"

#include "binary_ops.hpp"

int main()
{
	// experiment
	auto ta = crt_random_tensor<int32, CPU>({ 3, 4 }, 11);
	auto tb = crt_random_tensor<int32, CPU>({ 3, 4 }, 18);

	auto tc = tensor_add(ta, tb);

	std::cout << represent_tensor(ta) << std::endl;
	std::cout << represent_tensor(tb) << std::endl;
	std::cout << represent_tensor(tc) << std::endl;


	/*auto tensor = crt_random_tensor<float16, CPU>({ 3, 4 });
	auto cuda_tensor = tensor.copy_to_cuda();

	auto cpu_tensor = cuda_tensor.copy_to_host();

	std::cout << represent_tensor(cpu_tensor, 5);*/

	return 0;
}
