# This module is responsible for generating
# test cases for complex, fused operators.
# (complex building blocks for a network).
# Always the input tensors and output tensors are
# saved into separate files.

from dnn_inspector.dnninspect.tensor import save_tensor
from torch.nn import functional as F
import torch
import os

from os.path import join as pjoin


def generate_causal_conv1d_fwd_f32(path: str, test_name: str):
    """
        This operator is a special type of conv1d ops and also
        applies an activation.

        x shape: (batch, dims, seq_len)
        weights shape: (dims, width)
        bias shape: (dims)
        y shape: (batch, dims, seq_len)

        silu(conv1d(x, weights.unsqueeze(1), bias, stride=1, padding=width-1, groups=dims))[..., seq_len]
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    batch = 8
    dims = 2318
    seq_len = 638
    width = 4

    x = torch.randn([batch, dims, seq_len], dtype=torch.float32)
    w = torch.randn([dims, width], dtype=torch.float32)
    b = torch.randn([dims], dtype=torch.float32)

    # calculate the attention output
    y = F.silu(F.conv1d(x, w.unsqueeze(1), b, stride=1, padding=width-1, groups=dims))[:, :, :seq_len]

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
