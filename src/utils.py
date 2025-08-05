# In src/utils.py

import torch
import collections
from torch.nn import Module

def calculate_delta_parameters(pretrained_model: Module, finetuned_model: Module) -> collections.OrderedDict:
    """Calculates the delta, ensuring the output is always float."""
    pretrained_state_dict = pretrained_model.state_dict()
    finetuned_state_dict = finetuned_model.state_dict()
    delta_weights = collections.OrderedDict()

    for key in finetuned_state_dict.keys():
        # The root fix: convert both tensors to float BEFORE subtraction.
        delta = finetuned_state_dict[key].float() - pretrained_state_dict[key].float()
        delta_weights[key] = delta
        
    return delta_weights

# ... (The other functions in this file, calculate_parameters_size and calculate_compressed_size, remain the same) ...
def calculate_parameters_size(state_dict: collections.OrderedDict) -> float:
    total_bytes = 0
    for param in state_dict.values():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024 * 1024)

def calculate_compressed_size(compressed_data: dict) -> float:
    total_bytes = 0
    for layer_name, data in compressed_data.items():
        if data.get('is_compressed'):
            for patch in data['quantized_patches']:
                total_bytes += patch.numel()
            total_bytes += data['min_vals'].numel() * data['min_vals'].element_size()
            total_bytes += data['max_vals'].numel() * data['max_vals'].element_size()
            total_bytes += data['bit_allocations'].numel() * data['bit_allocations'].element_size()
        else:
            if 'uncompressed_delta' in data:
                tensor = data['uncompressed_delta']
                total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 * 1024)