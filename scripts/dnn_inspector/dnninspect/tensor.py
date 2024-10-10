# This module is responsible for providing functions
# to save a torch tensor into a byte format (.dat).

import functools
import struct
import torch
import sys

# Tensor data file format (no compression):
# dimension   - int (4 bytes)
# shape_1     - int (4 bytes)
# ...
# shape_dim   - int (4 bytes)
# data type   - int (4 bytes)
#   . INT8 - 0
#   . INT16 - 1
#   . INT32 - 2
#   . BFLOAT16 - 3
#   . FLOAT16 - 4
#   . FLOAT32 - 5
#   . FLOAT64 - 6
# tensor element 1 - data type (size depends on type)
# tensor element 2
# ...
# tensor element n (n can be calculated from shape)

# The file assumes default strides, no offset and default alignment.

def calc_num_elements(shape: torch.Size):
    n = 1
    for s in shape:
        n = n * s
    return n


def get_dtype_index(tensor: torch.Tensor):
    dtype_indices = {
        torch.int32: 2,
        torch.bfloat16: 3,
        torch.float16: 4,
        torch.float32: 5
    }
    
    index = None
    if tensor.dtype in dtype_indices:
        index = dtype_indices[tensor.dtype]
    return index


def get_torch_dtype(dtype: int):
    torch_dtypes = {
        2: torch.int32,     # int32
        3: torch.bfloat16,  # bfloat16
        4: torch.float16,   # float16
        5: torch.float32    # float32
    }
    torch_dtype = None
    if dtype in torch_dtypes:
        torch_dtype = torch_dtypes[dtype]
    return torch_dtype


def get_dtype_size(dtype: int):
    dtype_sizes = {
        2: 4,  # int32
        3: 2,  # bfloat16
        4: 2,  # float16
        5: 4   # float32
    }
    size = None
    if dtype in dtype_sizes:
        size = dtype_sizes[dtype]
    return size


def save_tensor(tensor: torch.Tensor, file_path: str):
    """
        Save the torch tensor into a dat file (no compression).

        tensor - Torch tensor to be saved.
        file_path - The path to the output file storing the tensor.
    """

    dim = tensor.dim()
    shape = tensor.size()

    dtype = get_dtype_index(tensor)
    assert(not(dtype is None))
    assert(dtype != 3)  # bfloat16 is not supported yet well

    with open(file_path, 'wb') as dat:

        # write the dim
        buffer = struct.pack("=i", dim)
        dat.write(buffer)

        # write the shape
        for s in shape:
            buffer = struct.pack("=i", s)
            dat.write(buffer)
        
        # write the data type
        buffer = struct.pack("=i", dtype)
        dat.write(buffer)

        # write the tensor elements
        data_tensor = tensor.flatten().cpu().detach().numpy()
        
        buffer = data_tensor.tobytes()
        dat.write(buffer)


def load_tensor(file_path: str) -> torch.Tensor:
    """
        Loads the torch tensor from a dat file (no compression).

        tensor - Torch tensor to be saved.
        file_path - The path to the output file storing the tensor.
    """
    int_as_bytes = struct.pack("=i", 0)
    int_size = len(int_as_bytes)
    
    tensor = None
    with open(file_path, 'rb') as dat:
        # read the dim
        dim = int.from_bytes(dat.read(int_size), sys.byteorder)

        # read the shape
        shape = list()
        for _ in range(dim):
            shape.append(int.from_bytes(dat.read(int_size), sys.byteorder))
        
        # read the data type
        dtype = int.from_bytes(dat.read(int_size), sys.byteorder)
        dtype_size = get_dtype_size(dtype)
        torch_dtype = get_torch_dtype(dtype)

        # read the data buffer
        buffer = dat.read(dtype_size * functools.reduce(lambda x, y: x * y, shape))
        tensor = torch.frombuffer(buffer, dtype=torch_dtype).clone().reshape(shape)

    return tensor
