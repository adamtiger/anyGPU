from sdp_test_cases import * 
from sf_test_cases import *

if __name__ == '__main__':

    path = r"C:\Data\AI\projects\anyGPU\artifacts"

    generate_sdp_fwd_nomask_noscore(path, "sdp_fwd_nomask_noscore_f32_16_64", 16, 64)
    generate_sdp_bwd_nomask_noscore(path, "sdp_bwd_nomask_noscore_f32_16_64", 16, 64)
    generate_softmax_bwd(path, "softmax_bwd_f32_16_64", 16, 64)

    generate_sf_file_reading(path, "sf_diffuser_tensors")
