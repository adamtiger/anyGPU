#include "utils.hpp"

int GlobalUUDGenerator::next_id = 0;

int GlobalUUDGenerator::generate_id()
{
	int temp = next_id;
	next_id += 1;
	return temp;
}
