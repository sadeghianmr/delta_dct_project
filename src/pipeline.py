import torch
import copy
from typing import Dict, Any, List, Tuple

from utils import calculate_delta_parameters
from core.compression import (
    tensor_to_patches,
    calculate_importance_scores,
    allocate_bit_widths,
    dct_and_quantize_patches
)
from core.decompression import (
    dequantize_and_idct_patches,
    patches_to_tensor,
    final_rescale
)

def compress_model(
    pretrained_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    patch_size: int,
    bit_strategy: List[Tuple[int, float]]
) -> Dict[str, Any]:
    """
    Runs the complete Delta-DCT compression pipeline on an entire model.
    It now handles non-2D tensors and sensitive final layers by passing them through uncompressed.
    """
    print("Starting model compression...")
    
    delta_weights = calculate_delta_parameters(pretrained_model, finetuned_model)
    compressed_model_data = {}
    
    for layer_name, delta_tensor in delta_weights.items():
        # --- FIX IS HERE ---
        # Add a check to avoid compressing the sensitive final classifier/pooler layers.
        is_compressible = (
            delta_tensor.dim() == 2 and 
            'classifier' not in layer_name and 
            'pooler' not in layer_name and 
            'head' not in layer_name and
            delta_tensor.numel() > patch_size**2
        )
        
        if is_compressible:
            print(f"Compressing layer: {layer_name}...")
            patches = tensor_to_patches(delta_tensor, patch_size)
            scores = calculate_importance_scores(patches)
            bits = allocate_bit_widths(scores, bit_strategy)
            quantized_patches, min_vals, max_vals = dct_and_quantize_patches(patches, bits)
            
            compressed_model_data[layer_name] = {
                'is_compressed': True,
                'quantized_patches': quantized_patches,
                'min_vals': min_vals,
                'max_vals': max_vals,
                'bit_allocations': bits,
                'original_shape': delta_tensor.shape,
                'patch_size': patch_size,
                'original_mean_abs': torch.mean(torch.abs(delta_tensor)).item()
            }
        else:
            # Otherwise, store the delta uncompressed.
            print(f"Skipping compression for layer '{layer_name}'. Storing uncompressed.")
            compressed_model_data[layer_name] = {
                'is_compressed': False,
                'uncompressed_delta': delta_tensor
            }

    print("\nModel compression finished.")
    return compressed_model_data

def decompress_model(
    pretrained_model: torch.nn.Module,
    compressed_data: Dict[str, Any]
) -> torch.nn.Module:
    """

    Reconstructs a fine-tuned model from compressed delta data.
    It handles both compressed and uncompressed deltas.
    """
    print("Starting model decompression...")
    reconstructed_model = copy.deepcopy(pretrained_model)
    
    for layer_name, layer_data in compressed_data.items():
        final_delta = None
        if layer_data['is_compressed']:
            print(f"Decompressing layer: {layer_name}...")
            reconstructed_patches = dequantize_and_idct_patches(
                layer_data['quantized_patches'], layer_data['min_vals'],
                layer_data['max_vals'], layer_data['bit_allocations']
            )
            reconstructed_delta = patches_to_tensor(
                reconstructed_patches, layer_data['original_shape'], layer_data['patch_size']
            )
            final_delta = final_rescale(reconstructed_delta, layer_data['original_mean_abs'])
        else:
            print(f"Applying uncompressed delta for layer: {layer_name}...")
            final_delta = layer_data['uncompressed_delta']
        
        with torch.no_grad():
            param = reconstructed_model.state_dict()[layer_name]
            target_device = param.device

            ## Fix
            param.data = param.data.float()  
            param.data += final_delta.to(target_device) # Reverted to simple addition

    print("\nModel decompression finished.")
    return reconstructed_model