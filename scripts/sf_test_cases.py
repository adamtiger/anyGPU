# This module is responsible for generating
# test cases for the safetensors data format
# reading.

from safetensors.numpy import load_file
from tensor import save_tensor
import torch
import os

from os.path import join as pjoin


def generate_sf_file_reading(path: str, test_name: str):

    # create test folder
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # 
    sf_file_path = r"C:\Data\AI\projects\anyGPU\artifacts\safetensors\diffusion_pytorch_model.safetensors"
    sf_tensors = load_file(sf_file_path)

    # save some of the tensors
    num_saved = 5
    step = len(sf_tensors) // num_saved

    cntr = 1
    for nm, t in sf_tensors.items():
        if cntr % step == 0:
            torch_t = torch.tensor(t)
            save_tensor(torch_t, pjoin(test_fld_name, f"{nm}.dat"))
        cntr += 1
    
    print(f"Generated: {test_name}")
