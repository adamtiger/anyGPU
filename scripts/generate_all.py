from sdp_test_cases import * 
from sf_test_cases import *
from ops_test_cases import *

if __name__ == '__main__':

    # path = r"C:\Data\AI\projects\anyGPU\artifacts"

    # generate_sdp_fwd_nomask_noscore(path, "test_sdp_fwd_nomask_noscore_f32_16_64", 16, 64)
    # generate_sdp_bwd_nomask_noscore(path, "test_sdp_bwd_nomask_noscore_f32_16_64", 16, 64)
    # generate_softmax_bwd(path, "test_softmax_bwd_f32_16_64", 16, 64)

    # generate_sf_file_reading(path, "test_sf_diffuser_tensors")

    # generate_layer_norm_fwd_f32(path, "test_layer_norm_fwd_f32")
    # generate_rms_norm_fwd_f32(path, "test_rms_norm_fwd_f32")
    # generate_silu_fwd_f32(path, "test_silu_fwd_f32")
    # generate_embedding_fwd_f32(path, "test_embedding_fwd_f32")
    # generate_rotary_embedding_fwd_f32(path, "test_rotary_embedding_fwd_f32")
    # generate_alt_rotary_embedding_fwd_f32(path, "test_alt_rotary_embedding_fwd_f32")
    # generate_linear_fwd_f32(path, "test_linear_fwd_f32")

    import torchvision
    import dnninspect as dnni
    m = torchvision.models.resnet18()
    #print(get_torch_module_info(m))
    #print(get_torch_top_submodule_names(m))
    #save_torch_module_weights(m, r"C:\Data\AI\projects\anyGPU\scripts\zamba2")

    x = torch.randn((1, 3, 224, 224), dtype=torch.float32)

    dnni.set_inspection_output_folder(r"C:\Data\AI\projects\anyGPU\scripts\zamba2")
    dnni.inspect_torch_module(m, "resnet18")(x)

    dnni.inspect_torch_module(m, "resnet18")(x)
