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

	Tensor<INT8, CPU> tensor({ 3, 5, 7 });

	std::cout << tensor.id << std::endl;
	std::cout << tensor.mem_buffer.id << std::endl;
	std::cout << tensor.mem_buffer.capacity << std::endl;

	return 0;
}
