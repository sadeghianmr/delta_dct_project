# In src/utils.py

import torch
import collections
from torch.nn import Module
import torch.nn.functional as F

# Standard JPEG Luminance Quantization Table
JPEG_QUANTIZATION_TABLE_8X8 = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=torch.float32)

def get_jpeg_quantization_table(patch_size: int) -> torch.Tensor:
    """
    Gets the standard JPEG quantization table, resized to the given patch size.

    Args:
        patch_size (int): The target edge size of the patch (e.g., 8, 16, 32).

    Returns:
        torch.Tensor: The resized quantization table.
    """
    if patch_size == 8:
        return JPEG_QUANTIZATION_TABLE_8X8
    table_reshaped = JPEG_QUANTIZATION_TABLE_8X8.view(1, 1, 8, 8)
    resized_table = F.interpolate(table_reshaped, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
    return resized_table.squeeze().clamp(min=1.0)


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