#include "add_vec.hpp"
#include <vector>
#include <random>
#include <iostream>

#include "tensor.hpp"


int main()
{
	// experiment
	int length = 1000;
	std::vector<float> ha(length);
	std::vector<float> hb(length);
	std::vector<float> hc(length);

	std::default_random_engine engine;
	std::uniform_real_distribution<float> uni_distr(1.f, 10.f);

	for (int ix = 0; ix < length; ++ix)
	{
		ha[ix] = uni_distr(engine);
		hb[ix] = uni_distr(engine);
	}

	//add_vectors(length, ha, hb, hc);

	// print
	for (int ix = 0; ix < 10; ++ix)
	{
		std::cout << ha[ix] << " " << hb[ix] << " " << hc[ix] << "\n";
	}

	auto tensor = crt_random_tensor<float16, CPU>({ 3, 4 });
	auto cuda_tensor = tensor.copy_to_cuda();

	auto cpu_tensor = cuda_tensor.copy_to_host();

	std::cout << represent_tensor(cpu_tensor, 5);

	return 0;
}
