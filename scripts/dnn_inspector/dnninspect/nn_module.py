# This module is responsible for registering
# and printing torch nn module related infos.
# It can also save the inputs and outputs to
# the networks.

import torch
import torch.nn as nn
import dnninspect.constants as const
from dnninspect.tensor import save_tensor
from os.path import join as pjoin
from typing import List
import json
import os

# the path to the folder storing the results of inspection
path_inspection_output_folder = "insepection_result"


def get_torch_module_info(m: nn.Module) -> dict:
    """
    Gathers the network block related information.
    Ignores the pytorch specific settings.
    """

    ignored_attributes = {
        "_parameters", 
        "_buffers",
        "_non_persistent_buffers_set",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_is_full_backward_hook",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_hooks_always_called",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_modules"
    }

    info = dict()
    info["name"] = m._get_name()
    attributes = m.__dict__
    for nm in attributes:
        if nm in ignored_attributes:
            continue
        info[nm] = str(attributes[nm])
    
    return info


def get_torch_top_submodule_names(m: nn.Module) -> List[str]:
    """
    Gathers the top level submodule names inside
    the given module. 
    """
    top_sub_mod_names = list()
    for sm in m.modules():
        if isinstance(sm, nn.Sequential):
            for ssm in sm:
                top_sub_mod_names.append(ssm._get_name())
        else:
            top_sub_mod_names.append(sm._get_name())
    return top_sub_mod_names


def save_torch_module_weights(m: nn.Module, weight_folder: str, prefix: str = None) -> None:
    """
    Saves all the weights of the module.

        prefix: to signal the concrete module (if there are more of the same module type)
    """
    prefix = (f"{prefix}." if prefix else "")
    for nm, pm in m.named_parameters(prefix=f"{prefix}{m._get_name().lower()}"):
        save_tensor(pm, pjoin(weight_folder, f"{nm}.dat"))


def save_torch_module_calc(m: nn.Module, ckp_folder: str, *inputs, **kwargs):
    """
    Saves the inputs and outputs of the module call (forward function).
        ckp_folder: the folder to save the model weights in
        *inputs: inputs enlisted
        **kwargs: key-value named arguments enlisted
    """
    # save inputs
    others = dict()
    for i, x in enumerate(inputs):
        nm = f"in_{i}"
        if isinstance(x, torch.Tensor):
            save_tensor(x, pjoin(ckp_folder, f"{nm}.dat"))
            others[f"{nm}_shape"] = x.size()
            others[f"{nm}_type"] = str(x.dtype)
        else:
            others[nm] = x
    
    for nm, x in kwargs.items():
        nm = f"in_{nm}"
        if isinstance(x, torch.Tensor):
            save_tensor(x, pjoin(ckp_folder, f"{nm}.dat"))
            others[f"{nm}_shape"] = x.size()
            others[f"{nm}_type"] = str(x.dtype)
        else:
            others[f"{nm}"] = x

    # execute the network
    outputs = m.forward(*inputs, **kwargs)

    # save outputs
    for i, x in enumerate(outputs):
        nm = f"out_{i}"
        if isinstance(x, torch.Tensor):
            save_tensor(x, pjoin(ckp_folder, f"{nm}.dat"))
            others[f"{nm}_shape"] = x.size()
            others[f"{nm}_type"] = str(x.dtype)
        else:
            others[f"{nm}"] = x
    
    # save others
    with open(pjoin(ckp_folder, "non_tensors.json"), "wt") as js:
        json.dump(others, js)
    return outputs


def set_inspection_output_folder(path: str):
    """
    Sets the output folder path for the module
    related data globally.
    """
    global path_inspection_output_folder
    path_inspection_output_folder = path


def inspect_torch_module(m: nn.Module, name: str) -> any:
    """
    Inspects a torch module. Returns a module executor function
    therefore the inspection and execution can be done in a single line.
    """
    # create output folder if needed
    output_path = pjoin(path_inspection_output_folder, name)
    save_module_data = not os.path.exists(output_path)

    module_weight_path = pjoin(output_path, const.MODULE_WEIGHTS)
    io_ckpt_path = pjoin(output_path, const.IO_CHECKPOINT)
    minfo_path = pjoin(output_path, const.MODULE_INFO_JSON)

    if save_module_data:
        os.mkdir(output_path)
        os.mkdir(module_weight_path)
        os.mkdir(io_ckpt_path)
    
    # save module info
    m_info = None
    if save_module_data:
        m_info = get_torch_module_info(m)
        m_info["top_modules"] = get_torch_top_submodule_names(m)
        save_torch_module_weights(m, module_weight_path, name)

    # create executor function
    # save input and output if needed
    def execute_module(*inputs, **kwargs):
        y = None
        if save_module_data:
            y = save_torch_module_calc(m, io_ckpt_path, *inputs, **kwargs)
        else:
            y = m.forward(*inputs, **kwargs)
        return y
    
    # save info into json
    if save_module_data:
        with open(minfo_path, "wt") as js:
            json.dump(m_info, js)

    return execute_module
