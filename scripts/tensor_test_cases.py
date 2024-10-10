# This module is responsible for testing
# the correctness of the tensor saver and loader.

from dnn_inspector.dnninspect.tensor import load_tensor
from dnn_inspector.dnninspect.tensor import save_tensor
from os.path import join as pjoin
import torch


def test_tensor_io(path: str, torch_dtype):
    # generate random inputs
    m = 8
    n = 4

    x = (torch.randn([m, n], dtype=torch.float32) * 100.0).to(dtype=torch_dtype)

    # save tensors
    x_path = pjoin(path, "x.dat")
    save_tensor(x, x_path)
    y = load_tensor(x_path)

    print(f"IO test for {torch_dtype}: {torch.allclose(x, y)}")


if __name__ == '__main__':
    path = r"C:\Data\AI\projects\anyGPU\artifacts"
    test_tensor_io(path, torch.float32)
    test_tensor_io(path, torch.float16)
    test_tensor_io(path, torch.int32)
