import torch
import collections
from torch.nn import Module
import torch.nn.functional as F

def calculate_delta_parameters(pretrained_model: Module, finetuned_model: Module) -> collections.OrderedDict:
    """
    Calculates the difference (delta) between the parameters of a fine-tuned model
    and a pre-trained model, ensuring the output is always float.

    Args:
        pretrained_model (torch.nn.Module): The original pre-trained model.
        finetuned_model (torch.nn.Module): The model after fine-tuning on a specific task.

    Returns:
        collections.OrderedDict: A dictionary where keys are layer names and values are the
                                 float delta parameter tensors for each layer.
    """
    pretrained_state_dict = pretrained_model.state_dict()
    finetuned_state_dict = finetuned_model.state_dict()
    delta_weights = collections.OrderedDict()

    for key in finetuned_state_dict.keys():
        # The root fix: convert both tensors to float BEFORE subtraction.
        delta = finetuned_state_dict[key].float() - pretrained_state_dict[key].float()
        delta_weights[key] = delta
        
    return delta_weights

def calculate_parameters_size(state_dict: collections.OrderedDict) -> float:
    """
    Calculates the total size of a model's state_dict in megabytes (MB).

    Args:
        state_dict (collections.OrderedDict): The state dictionary of a model or delta weights.

    Returns:
        float: The total size of the parameters in megabytes.
    """
    total_bytes = 0
    for param in state_dict.values():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024 * 1024)

def calculate_compressed_size(compressed_data: dict) -> float:
    """
    Calculates the total size of the compressed data structure in megabytes (MB).
    This version correctly handles both compressed and uncompressed layers.

    Args:
        compressed_data (dict): The main dictionary containing all compressed layer data.

    Returns:
        float: The total size of the compressed data in megabytes.
    """
    total_bytes = 0
    for layer_name, data in compressed_data.items():
        if data.get('is_compressed'):
            # Size of quantized patches
            # The internal structure for DWT is a list of lists, so we handle it.
            if isinstance(data['quantized_patches'][0], list): # DWT case
                for patch_coeffs in data['quantized_patches']:
                    for coeff_matrix in patch_coeffs:
                        total_bytes += coeff_matrix.numel() # int8 is 1 byte
            else: # DCT case
                for patch in data['quantized_patches']:
                    total_bytes += patch.numel() # int8 is 1 byte
            
            # Size of metadata (floats and ints)
            total_bytes += data['min_vals'].numel() * data['min_vals'].element_size()
            total_bytes += data['max_vals'].numel() * data['max_vals'].element_size()
            total_bytes += data['bit_allocations'].numel() * data['bit_allocations'].element_size()
        else:
            # If not compressed, calculate the size of the stored uncompressed delta.
            if 'uncompressed_delta' in data:
                tensor = data['uncompressed_delta']
                total_bytes += tensor.numel() * tensor.element_size()
    
    return total_bytes / (1024 * 1024)

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
    resized_table = F.interpolate(
        table_reshaped,
        size=(patch_size, patch_size),
        mode='bilinear',
        align_corners=False
    )
    return resized_table.squeeze().clamp(min=1.0)