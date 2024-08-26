# This module is responsible for generating
# test cases for attention mechanism variants.
# Always the input tensors and output tensors are
# saved into separate files.

from tensor import save_tensor
from torch.nn import functional as F
import torch
import os

from os.path import join as pjoin


def generate_sdp_fwd_nomask_noscore(path: str, test_name: str, N: int, d: int):
    """
        Single dot product (attention). Simple case, 
        no masking and no score.    
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
        N: Number of embeddings.
        d: Embedding size.
    """

    # generate random q, k and v
    tensor_size = (1, 1, N, d)
    q = torch.randn(tensor_size, dtype=torch.float32)
    k = torch.randn(tensor_size, dtype=torch.float32)
    v = torch.randn(tensor_size, dtype=torch.float32)
    
    # calculate the attention output
    y = F.scaled_dot_product_attention(q, k, v)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # remove first 2 dimensions (they are placeholders)
    q = q.squeeze()
    k = k.squeeze()
    v = v.squeeze()
    y = y.squeeze()

    # save tensors
    save_tensor(q, pjoin(test_fld_name, "q.dat"))
    save_tensor(k, pjoin(test_fld_name, "k.dat"))
    save_tensor(v, pjoin(test_fld_name, "v.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"q: {q[0, 0:10]}")
    print(f"y: {y[0, 0:10]}")


def generate_sdp_bwd_nomask_noscore(path: str, test_name: str, N: int, d: int):
    """
        Single dot product backward (attention). Simple case, 
        no masking and no score.

        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
        N: Number of embeddings.
        d: Embedding size.
    """

    # generate random q, k and v
    tensor_size = (1, 1, N, d)
    q = torch.randn(tensor_size, dtype=torch.float32, requires_grad=True)
    k = torch.randn(tensor_size, dtype=torch.float32, requires_grad=True)
    v = torch.randn(tensor_size, dtype=torch.float32, requires_grad=True)
    
    # calculate the attention output
    y = F.scaled_dot_product_attention(q, k, v)

    # calculate the gradients
    y.backward(torch.ones(tensor_size, dtype=torch.float32))

    grad_q = q.grad
    grad_k = k.grad
    grad_v = v.grad

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # remove first 2 dimensions (they are placeholders)
    q = q.squeeze()
    k = k.squeeze()
    v = v.squeeze()
    
    grad_q = grad_q.squeeze()
    grad_k = grad_k.squeeze()
    grad_v = grad_v.squeeze()

    # save tensors
    save_tensor(q, pjoin(test_fld_name, "q.dat"))
    save_tensor(k, pjoin(test_fld_name, "k.dat"))
    save_tensor(v, pjoin(test_fld_name, "v.dat"))

    save_tensor(grad_q, pjoin(test_fld_name, "grad_q.dat"))
    save_tensor(grad_k, pjoin(test_fld_name, "grad_k.dat"))
    save_tensor(grad_v, pjoin(test_fld_name, "grad_v.dat"))

    # print sample
    print(f"q: {q[0, 0:10]}")
    print(f"grad_q: {grad_q[0, 0:10]}")


if __name__ == '__main__':
    path = r"C:\Data\AI\projects\anyGPU\artifacts"
    generate_sdp_fwd_nomask_noscore(path, "sdp_fwd_nomask_noscore_f32_16_64", 16, 64)
    generate_sdp_bwd_nomask_noscore(path, "sdp_bwd_nomask_noscore_f32_16_64", 16, 64)
