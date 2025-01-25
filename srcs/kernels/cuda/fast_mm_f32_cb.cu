#include "fast_mm.cuh"
#include "cublas_v2.h"
#include "transp_ops.hpp"

// refs: https://docs.nvidia.com/cuda/cublas/#cublas-level-3-function-reference
//       https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/gemm/cublas_gemm_example.cu

void cu_fast_mm_f32_cb(
	const Tensor<float32, CUDA>& xt,  // input (multiple of 128)
	const Tensor<float32, CUDA>& wt,  // weight (multiple of 128)
	Tensor<float32, CUDA>& yt)
{
	const float alpha = 1.f;
	const float beta = 0.f;

	const int m = xt.shape[0];
	const int n = wt.shape[1];
	const int k = xt.shape[1];

	const int lda = k;  // anygpu -> row-major, cublas -> column-major
	const int ldb = n;
	const int ldc = m;

	cublasHandle_t handle;
	cublasCreate(&handle);

	auto success = cublasSgemm(
		handle, 
		CUBLAS_OP_T, 
		CUBLAS_OP_T, 
		m, n, k, 
		&alpha, 
		xt.buffer(), 
		lda, 
		wt.buffer(), 
		ldb, 
		&beta, 
		yt.buffer(), 
		ldc
	);

	cublasDestroy(handle);
	CUDA_CHECK_LAST_ERROR();

	std::swap(yt.shape[0], yt.shape[1]);  // anygpu -> row-major, cublas -> column-major
	yt = tensor_transp(yt);
}
