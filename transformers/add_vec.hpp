#pragma once

#include <vector>

using Vector = std::vector<float>;

void add_vectors(
	const int length, 
	const Vector& ha, 
	const Vector& hb,
	Vector& hc
);
