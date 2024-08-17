#ifndef __UTILS__
#define __UTILS__

#include "datatypes.hpp"

/*
  Generates global universal unique ids.
*/
class GlobalUUIDGenerator
{
public:
	static int generate_id();

private:
	static int next_id;
};

// helper functions for tensor size calculations

/*
  If the alignment is the default for the given data type,
  this function returns the number of elements in the tensor.
  Default alignment means the byte size of 1 tensor element.
*/
int calc_default_size(const std::vector<int>& shape);

/*
  If the alignment is the default for the given data type,
  this function returns the number of elements in the tensor.
  Default alignment means the byte size of 1 tensor element.
*/
int calc_default_size(const int dim, const DimArray& shape);

/*
  Calculates the default stride from the given shape.
*/
std::vector<int> calc_default_stride(const std::vector<int>& shape);

/*
  Transforms int vector to array.
  Vector can be shape or stride.
*/
DimArray cvt_vector2array(const std::vector<int>& v);

/*
  Dimension from vector.
*/
int calc_dim(const std::vector<int>& v);

#endif  // __UTILS__
