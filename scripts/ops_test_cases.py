# This module is responsible for generating
# test cases for the basic operators 
# (building blocks for the network).
# Always the input tensors and output tensors are
# saved into separate files.

from torchtune.modules import RotaryPositionalEmbeddings
from dnn_inspector.dnninspect.tensor import save_tensor
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


def generate_gelu_fwd_f32(path: str, test_name: str):
    """
        Gelu test.
        ref: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    tensor_size = (20, 30, 16, 104)
    x = torch.randn(tensor_size, dtype=torch.float32)

    # calculate the attention output
    y = F.gelu(x, approximate='none')

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x, pjoin(test_fld_name, "x.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_embedding_fwd_f32(path: str, test_name: str):
    """
        Embedding test.
        ref: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#embedding
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    indices = torch.randint(0, 117, (12, 4), dtype=torch.int32)
    embedding = torch.randn((118, 71), dtype=torch.float32)

    # calculate the attention output
    y = F.embedding(indices, embedding)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(indices, pjoin(test_fld_name, "indices.dat"))
    save_tensor(embedding, pjoin(test_fld_name, "embedding.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_rotary_embedding_fwd_f32(path: str, test_name: str):
    """
        Embedding test.
        ref: https://pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html#rotarypositionalembeddings
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    max_seq_len = 1024

    batch = 2
    seq = max_seq_len // 2
    heads = 4
    dim = 64
    q = torch.randn((batch, seq, heads, dim), dtype=torch.float32)
    pos_idcs = torch.randint(0, max_seq_len, (batch, seq), dtype=torch.int32)

    # calculate the attention output
    rope = RotaryPositionalEmbeddings(dim, max_seq_len)
    y = rope(q, input_pos=pos_idcs)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(q, pjoin(test_fld_name, "q.dat"))
    save_tensor(pos_idcs, pjoin(test_fld_name, "p.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_alt_rotary_embedding_fwd_f32(path: str, test_name: str):
    """
        Alternative Rotary Embedding test.
        ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py#L131
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """

    # implementation
    def alternative_rotary_embedding(q, position_ids, dim, base=10000):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        unsqueeze_dim=2
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        return q_embed


    # generate random inputs
    batch = 4
    seq_len = 512
    heads = 2
    dim = 64
    q = torch.randn((batch, seq_len, heads, dim), dtype=torch.float32)
    pos_ids = torch.randint(0, seq_len * 2, (batch, seq_len), dtype=torch.int32)

    # calculate the attention output
    y = alternative_rotary_embedding(q, pos_ids, dim)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(q, pjoin(test_fld_name, "q.dat"))
    save_tensor(pos_ids, pjoin(test_fld_name, "p.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_linear_fwd_f32(path: str, test_name: str):
    """
        Linear test.
        ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html#torch.nn.functional.linear
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    tensor_size = (20, 2, 12, 328)
    x = torch.randn(tensor_size, dtype=torch.float32)
    w = torch.randn((328, 120), dtype=torch.float32)
    b = torch.randn((120,), dtype=torch.float32)

    # calculate the attention output
    y = F.linear(x, w.T, b)

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


def generate_transpose_fwd_f32(path: str, test_name: str):
    """
        Transpose test.
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    tensor_size = (20, 4, 12, 128)
    x = torch.randn(tensor_size, dtype=torch.float32)

    # calculate the attention output
    y = x.transpose(1, 2)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x, pjoin(test_fld_name, "x.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")


def generate_concat_fwd_f32(path: str, test_name: str):
    """
        Concatenation test.
    
        path: The path to a folder where the test case folder will be stored. 
        test_name: The name of the folder.
    """
    # generate random inputs
    ts_1 = (20, 4, 12, 256)
    ts_2 = (20, 4, 12, 512)
    x1 = torch.randn(ts_1, dtype=torch.float32)
    x2 = torch.randn(ts_2, dtype=torch.float32)

    # calculate the attention output
    y = torch.concatenate([x1, x2], dim=-1)

    # create test folders
    test_fld_name = pjoin(path, test_name)
    os.mkdir(test_fld_name)

    # save tensors
    save_tensor(x1, pjoin(test_fld_name, "x1.dat"))
    save_tensor(x2, pjoin(test_fld_name, "x2.dat"))
    save_tensor(y, pjoin(test_fld_name, "y.dat"))

    # print sample
    print(f"Generated: {test_name}")
