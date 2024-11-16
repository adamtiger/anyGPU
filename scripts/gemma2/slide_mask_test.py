from dnn_inspector.dnninspect.tensor import save_tensor
import torch
import os

from os.path import join as pjoin


def generate_slide_mask_f32(path: str, test_name: str):
    """    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    sliding_window = 34
    tensor_size = (5, 1, 121, 41)
    attention_mask = torch.randn(tensor_size, dtype=torch.float32)

    min_dtype = torch.finfo(torch.float32).min
    sliding_window_mask = torch.tril(
        torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-sliding_window
    )
    y = torch.where(sliding_window_mask, min_dtype, attention_mask)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(attention_mask, pjoin(test_fld_name, "attention_mask.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")
