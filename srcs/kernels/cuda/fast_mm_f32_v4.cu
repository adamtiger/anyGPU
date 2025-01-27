#include "fast_mm.cuh"

const int WARP_SIZE = 32;

// register level tile size
//   per-thread tile
const int RH = 4;
const int RW = 4;
const int RK = 4;

// warp organization inside a warp tile
const int NT_H = 4;  // num threads in warp (horizontal)
const int NT_W = 8;  // num threads in warp (vertical)
static_assert(NT_H* NT_W == WARP_SIZE);

// warp level tile size (in output)
const int WH = NT_H * RH;
const int WW = NT_W * RW;
const int WK = RK;

// shared mem level tile size
//   per-block tile
const int NW_H = 4;
const int NW_W = 2;

const int TH = NW_H * WH;
const int TW = NW_W * WW;
const int TK = WARP_SIZE * 2;


__global__ void cu_fast_mm_f32_v4_kernel(
	const int x_width,
	const int w_width,
	const float32* dx,
	const float32* dw,
	float32* dy)
{
	// TODO: ...
}


void cu_fast_mm_f32_v4(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA>& wt,
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int x_height = xt.shape[0];
	const int x_width = xt.shape[1];
	const int w_width = wt.shape[1];

	auto* dx = xt.buffer();
	auto* dw = wt.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	/*dim3 bs = { BS, BS, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, TS_W), calc_req_num_blocks(x_height, TS_H), 1 };

	cu_fast_mm_f32_v4_kernel<<<gs, bs>>>(x_width, w_width, dx, dw, dy);*/
	CUDA_CHECK_LAST_ERROR();
}
