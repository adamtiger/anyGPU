#include "add_vec.hpp"
#include <cuda_runtime.h>
#include <cassert>

__global__ void add_vec_kernel(const int length, float* da, float* db, float* dc)
{
	int tix = threadIdx.x;

	if (tix < length)
	{ 
		dc[tix] = da[tix] + db[tix];
	}
}

void add_vectors(
	const int length,
	const Vector& ha,
	const Vector& hb,
	Vector& hc)
{
	assert(length <= 1024);

	// allocate memories on the device
	int buffer_size = length * sizeof(float);

	float *da, *db, *dc;
	cudaMalloc(&da, buffer_size);
	cudaMalloc(&db, buffer_size);
	cudaMalloc(&dc, buffer_size);

	// move memory from host to device
	cudaMemcpy(da, ha.data(), buffer_size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb.data(), buffer_size, cudaMemcpyHostToDevice);

	// execute kernel
	dim3 gs(1, 1, 1);
	dim3 bs(1024, 1, 1);
	add_vec_kernel<<<gs, bs>>>(length, da, db, dc);

	// move data back
	cudaMemcpy(hc.data(), dc, buffer_size, cudaMemcpyDeviceToHost);

	// clean up
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
}
