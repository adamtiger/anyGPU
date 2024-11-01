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


def save_function_calc(func: any, ckp_folder: str, dict_out: bool, *inputs, **kwargs):
    """
    Saves the inputs and outputs of the function call.
        ckp_folder: the folder to save the model weights in
        *inputs: inputs enlisted
        **kwargs: key-value named arguments enlisted
    """
    # save inputs
    others = dict()
    for i, x in enumerate(inputs):
        nm = f"in_{i}"
        if isinstance(x, torch.Tensor):

            if x.dtype == torch.long:  # long is not supported yet
                x = x.to(dtype=torch.int32)
            elif x.dtype == torch.bfloat16:
                x = x.to(dtype=torch.float32)

            save_tensor(x, pjoin(ckp_folder, f"{nm}.dat"))
            others[f"{nm}_shape"] = x.size()
            others[f"{nm}_type"] = str(x.dtype)
        else:
            others[nm] = str(x)
    
    for nm, x in kwargs.items():
        nm = f"in_{nm}"
        if isinstance(x, torch.Tensor):

            if x.dtype == torch.int64:  # (int64) long is not supported yet
                x = x.to(dtype=torch.int32)
            elif x.dtype == torch.bfloat16:
                x = x.to(dtype=torch.float32)

            save_tensor(x, pjoin(ckp_folder, f"{nm}.dat"))
            others[f"{nm}_shape"] = x.size()
            others[f"{nm}_type"] = str(x.dtype)
        else:
            others[f"{nm}"] = str(x)

    # execute the network
    outputs = func(*inputs, **kwargs)

    # save outputs
    iterable_outputs = outputs
    if isinstance(outputs, torch.Tensor):
        iterable_outputs = [outputs]  # create a list from it
    elif dict_out:
        iterable_outputs = outputs.values()

    for i, y in enumerate(iterable_outputs):
        nm = f"out_{i}"
        if isinstance(y, torch.Tensor):

            if y.dtype == torch.int64:  # (int64) long is not supported yet
                y = y.to(dtype=torch.int32)
            elif y.dtype == torch.bfloat16:
                y = y.to(dtype=torch.float32)

            save_tensor(y, pjoin(ckp_folder, f"{nm}.dat"))
            others[f"{nm}_shape"] = y.size()
            others[f"{nm}_type"] = str(y.dtype)
        else:
            others[f"{nm}"] = str(y)
    
    # save others
    with open(pjoin(ckp_folder, "non_tensors.json"), "wt") as js:
        json.dump(others, js)
    return outputs


def save_torch_module_calc(m: nn.Module, ckp_folder: str, dict_out: bool, *inputs, **kwargs):
    """
    Saves the inputs and outputs of the module call (forward function).
        ckp_folder: the folder to save the model weights in
        *inputs: inputs enlisted
        **kwargs: key-value named arguments enlisted
    """
    return save_function_calc(m.forward, ckp_folder, dict_out, *inputs, **kwargs)


def set_inspection_output_folder(path: str):
    """
    Sets the output folder path for the module
    related data globally.
    """
    global path_inspection_output_folder
    path_inspection_output_folder = path


def inspect_torch_module(m: nn.Module, name: str, dict_out: bool = False) -> any:
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
            y = save_torch_module_calc(m, io_ckpt_path, dict_out, *inputs, **kwargs)
        else:
            y = m.forward(*inputs, **kwargs)
        return y
    
    # save info into json
    if save_module_data:
        with open(minfo_path, "wt") as js:
            json.dump(m_info, js)

    return execute_module


def inspect_torch_tensor(t: torch.Tensor, location: str, name: str) -> any:
    """
    Inspects and saves a torch tensor.
    The tensor with this name is saved only once.
        location: name of the location
        name: tensor name
    """
    # create output folder if needed
    output_path = pjoin(path_inspection_output_folder, location)

    tensor_fld_path = pjoin(output_path, const.TENSORS)
    tensor_path = pjoin(tensor_fld_path, f"{name}.dat")
    tinfo_path = pjoin(tensor_fld_path, f"tinfo_{name}.json")

    save_tensor_data = not os.path.exists(tensor_path)

    if save_tensor_data:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(tensor_fld_path):
            os.mkdir(tensor_fld_path)
    
    # save module info
    m_info = dict()
    if save_tensor_data:
        save_tensor(t, tensor_path)
        m_info["shape"] = t.size()
        m_info["type"] = str(t.dtype)

    # save info into json
    if save_tensor_data:
        with open(tinfo_path, "wt") as js:
            json.dump(m_info, js)


def inspect_function(func: any, name: str, dict_out: bool = False) -> any:
    """
    Inspects a function. Returns a function executor function
    therefore the inspection and execution can be done in a single line.
    """
    # create output folder if needed
    output_path = pjoin(path_inspection_output_folder, name)
    save_function_call_data = not os.path.exists(output_path)

    if save_function_call_data:
        os.mkdir(output_path)
    
    # create executor function
    # save input and output if needed
    def execute_module(*inputs, **kwargs):
        y = None
        if save_function_call_data:
            y = save_function_calc(func, output_path, dict_out, *inputs, **kwargs)
        else:
            y = func(*inputs, **kwargs)
        return y

    return execute_module


def inspect_function_repeated(func: any, name: str, reps: int, dict_out: bool = False) -> any:
    """
    Inspects a function. Returns a function executor function
    therefore the inspection and execution can be done in a single line.

    The results are saved during several (reps) consecutive visits. 
    """
    # create output folder if needed
    output_path = pjoin(path_inspection_output_folder, name)
    is_main_not_folder_exist = not os.path.exists(output_path)

    if is_main_not_folder_exist:
        os.mkdir(output_path)
    
    # create the next checkpoint folder name
    num_saved = len(os.listdir(output_path))
    save_function_call_data = num_saved < reps

    save_path = None
    if save_function_call_data:
        save_path = pjoin(output_path, f"{const.IO_CHECKPOINT}_{num_saved + 1}")
        os.mkdir(save_path)

    # create executor function
    # save input and output if needed
    def execute_module(*inputs, **kwargs):
        y = None
        if save_function_call_data:
            y = save_function_calc(func, save_path, dict_out, *inputs, **kwargs)
        else:
            y = func(*inputs, **kwargs)
        return y

    return execute_module
