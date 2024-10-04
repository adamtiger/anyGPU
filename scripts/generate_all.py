from sdp_test_cases import * 
from sf_test_cases import *
from ops_test_cases import *

if __name__ == '__main__':

    path = r"C:\Data\AI\projects\anyGPU\artifacts"

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
    generate_transpose_fwd_f32(path, "test_transpose_fwd_f32")


    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # import torch

    # tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-1.2B")
    # model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba2-1.2B", device_map="cuda", torch_dtype=torch.bfloat16)

    # input_text = "What factors contributed to the fall of the Roman Empire?"
    # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    # outputs = model.generate(**input_ids, max_new_tokens=100)
    # print(tokenizer.decode(outputs[0]))

