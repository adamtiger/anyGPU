# This module is responsible for providing functions
# to save a torch tensor into a byte format (.dat).

import struct
import torch

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


def save_tensor(tensor: torch.Tensor, file_path: str):
    """
        Save the torch tensor into a dat file (no compression).

        tensor - Torch tensor to be saved.
        file_path - The path to the output file storing the tensor.
    """

    dim = tensor.dim()
    shape = tensor.size()

    assert(tensor.dtype == torch.float32)
    dtype = 5  # assume float32 for now

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
        num_elements = calc_num_elements(shape)
        data_tensor = tensor.flatten().detach().numpy()
        
        buffer = data_tensor.tobytes()
        # for i in range(1, num_elements):
        #     buffer = buffer + struct.pack("=f", data_tensor[i])  # it is float
        dat.write(buffer)
