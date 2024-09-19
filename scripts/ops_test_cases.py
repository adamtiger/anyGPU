# This module is responsible for generating
# test cases for the basic operators 
# (building blocks for the network).
# Always the input tensors and output tensors are
# saved into separate files.

from tensor import save_tensor
from torch.nn import functional as F
import torch
import os

from os.path import join as pjoin


def generate_layer_norm_fwd_f32(path: str, test_name: str):
    """
        Layer normalization test.
        ref: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#layernorm  
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    tensor_size = (20, 30, 12, 84)
    norm_size = (12, 84)
    x = torch.randn(tensor_size, dtype=torch.float32)
    w = torch.randn(norm_size, dtype=torch.float32)
    b = torch.randn(norm_size, dtype=torch.float32)
    eps = 2e-3

    # calculate the attention output
    y = F.layer_norm(x, norm_size, w, b, eps)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x, pjoin(test_fld_name, "x.dat"))
    save_tensor(w, pjoin(test_fld_name, "w.dat"))
    save_tensor(b, pjoin(test_fld_name, "b.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_rms_norm_fwd_f32(path: str, test_name: str):
    """
        RMS normalization test.
        ref: https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html#rmsnorm
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    tensor_size = (20, 30, 15, 88)
    norm_size = (15, 88)
    x = torch.randn(tensor_size, dtype=torch.float32)
    w = torch.randn(norm_size, dtype=torch.float32)
    eps = 2e-3

    # calculate the attention output
    y = F.rms_norm(x, norm_size, w, eps)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x, pjoin(test_fld_name, "x.dat"))
    save_tensor(w, pjoin(test_fld_name, "w.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_silu_fwd_f32(path: str, test_name: str):
    """
        Silu test.
        ref: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#silu
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    tensor_size = (20, 30, 12, 104)
    x = torch.randn(tensor_size, dtype=torch.float32)

    # calculate the attention output
    y = F.silu(x)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x, pjoin(test_fld_name, "x.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")
