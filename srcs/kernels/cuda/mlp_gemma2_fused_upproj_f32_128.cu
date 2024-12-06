#include "mlp_gemma2_fused_upproj.cuh"

constexpr int x_width = 2304;
constexpr int w_width = 9216;

constexpr int NUM_WARPS = 4;
constexpr int WARP_SIZE = 32;

constexpr int TS_K = 16;
constexpr int TS_W = WARP_SIZE * NUM_WARPS;
constexpr int TS_H = 16;  // tile size for x height TODO: ???


__global__ void cu_mlp_gemma2_fused_uprpoj_f32_128_kernel(
	const int sl,
	const float32* dx,
	const float32* dwgp,
	const float32* dwup,
	float32* dy)
{

}


void cu_mlp_gemma2_fused_uprpoj_f32_128(
	const Tensor<float32, CUDA>& xt,     // input (batch * seq_len, 2304)
	const Tensor<float32, CUDA>& wt_gp,  // gate proj weight, (2304, 9216)
	const Tensor<float32, CUDA>& wt_up,  // gate up proj weight, (2304, 9216) 
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int sl = xt.shape[0] * xt.shape[1];

	auto* dx = xt.buffer();
	auto* dwgp = wt_gp.buffer();
	auto* dwup = wt_up.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { WARP_SIZE * NUM_WARPS, 1, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, bs.x), 1, 1 };  // TODO: seq len can grow with iteration!

	cu_mlp_gemma2_fused_uprpoj_f32_128_kernel<<<gs, bs>>>(sl, dx, dw_gp, dw_up, dy);
	CUDA_CHECK_LAST_ERROR();
}
