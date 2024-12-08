from dnn_inspector.dnninspect.tensor import save_tensor
from torch.nn import functional as F
import torch
import os

from os.path import join as pjoin


def generate_fused_mlp_upproj_f32(path: str, test_name: str, seq_len: int = 9):
    """    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    hidden_size = 2304
    upp_hidden_size = hidden_size * 4
    tensor_size = (1, seq_len, hidden_size)
    x = torch.randn(tensor_size, dtype=torch.float32)
    w_gate_proj = torch.randn((hidden_size, upp_hidden_size), dtype=torch.float32)
    w_up_proj = torch.randn((hidden_size, upp_hidden_size), dtype=torch.float32)
    w_down_proj = torch.randn((upp_hidden_size, hidden_size), dtype=torch.float32)

    x_gated = F.gelu(torch.matmul(x, w_gate_proj), approximate='tanh')
    x_up = torch.matmul(x, w_up_proj)
    z = x_gated * x_up
    y = torch.matmul(z, w_down_proj)

    # create test folders
    test_fld_name = pjoin(path, f"{test_name}_sqlen{seq_len}")
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x, pjoin(test_fld_name, "x.dat"))
    save_tensor(w_gate_proj, pjoin(test_fld_name, "w_gate_proj.dat"))
    save_tensor(w_up_proj, pjoin(test_fld_name, "w_up_proj.dat"))
    save_tensor(w_down_proj, pjoin(test_fld_name, "w_down_proj.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")
