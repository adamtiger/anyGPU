#include "mlp_gemma2_dp_linear.cuh"

constexpr int x_width = 9216;
constexpr int w_width = 2304;

constexpr int NUM_WARPS = 8;
constexpr int WARP_SIZE = 32;

constexpr int TS_H = 32;
constexpr int TS_K = 64;
constexpr int TS_W = 32;


__global__ void cu_mlp_gemma2_dp_linear_f32_v1_kernel(
	const int sl,
	const float32* dx,
	const float32* dwdp,
	float32* dy)
{

}


void cu_mlp_gemma2_dp_linear_f32_v1(
	const Tensor<float32, CUDA>& xt,     // input (batch * seq_len, 9216)
	const Tensor<float32, CUDA>& wt_dp,  // down proj weight, (9216, 2304)
	Tensor<float32, CUDA>& yt)
{
	// calc kernel arguments
	const int sl = xt.shape[0] * xt.shape[1];

	auto* dx = xt.buffer();
	auto* dwdp = wt_dp.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { WARP_SIZE, NUM_WARPS, 1 };
	dim3 gs = { calc_req_num_blocks(w_width, TS_W), calc_req_num_blocks(sl, TS_H), 1 };

	cu_mlp_gemma2_dp_linear_f32_v1_kernel<<<gs, bs>>>(sl, dx, dwdp, dy);
	CUDA_CHECK_LAST_ERROR();
}
