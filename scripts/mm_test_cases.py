from dnn_inspector.dnninspect.tensor import save_tensor
from torch.nn import functional as F
import torch
import os

from os.path import join as pjoin


def generate_mm_fwd_f32(path: str):
    test_name = "test_mm_m1024_n2048_k2304_f32"
    # generate random inputs
    tensor_size = (1024, 2304)
    norm_size = (2304, 2048)
    x = torch.rand(tensor_size, dtype=torch.float32)
    w = torch.rand(norm_size, dtype=torch.float32)

    # calculate the attention output
    y = torch.matmul(x, w)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x, pjoin(test_fld_name, "x.dat"))
    save_tensor(w, pjoin(test_fld_name, "w.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_mm_fwd_f16(path: str):
    test_name = "test_mm_m1024_n2048_k2304_f16"
    # generate random inputs
    tensor_size = (1024, 2304)
    norm_size = (2304, 2048)
    x = torch.rand(tensor_size, dtype=torch.float16)
    w = torch.rand(norm_size, dtype=torch.float16)

    # calculate the attention output
    y = torch.matmul(x, w)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x, pjoin(test_fld_name, "x.dat"))
    save_tensor(w, pjoin(test_fld_name, "w.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_mm_big_fwd_f32(path: str):
    test_name = "test_mm_m4608_n9216_k2304_f32"
    # generate random inputs
    tensor_size = (4608, 2304)
    norm_size = (2304, 9216)
    x = torch.rand(tensor_size, dtype=torch.float32)
    w = torch.rand(norm_size, dtype=torch.float32)

    # calculate the attention output
    y = torch.matmul(x, w)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x, pjoin(test_fld_name, "x.dat"))
    save_tensor(w, pjoin(test_fld_name, "w.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_mm_big_fwd_f16(path: str):
    test_name = "test_mm_m4608_n9216_k2304_f16"
    # generate random inputs
    tensor_size = (4608, 2304)
    norm_size = (2304, 9216)
    x = torch.rand(tensor_size, dtype=torch.float16)
    w = torch.rand(norm_size, dtype=torch.float16)

    # calculate the attention output
    y = torch.matmul(x, w)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x, pjoin(test_fld_name, "x.dat"))
    save_tensor(w, pjoin(test_fld_name, "w.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_all_mm(path: str):
    generate_mm_fwd_f32(path)
    generate_mm_fwd_f16(path)
    generate_mm_big_fwd_f32(path)
    generate_mm_big_fwd_f16(path)
