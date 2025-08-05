import torch
import copy
from typing import Dict, Any, List, Tuple
from torch.nn import Module

# Import all necessary building blocks
from utils import calculate_delta_parameters
from core.compression import (
    tensor_to_patches,
    calculate_importance_scores,
    allocate_bit_widths,
    dct_and_quantize_patches,
    dwt_and_quantize_patches  # Import the new DWT function
)
from core.decompression import (
    dequantize_and_idct_patches,
    dequantize_and_idwt_patches, # Import the new IDWT function
    patches_to_tensor,
    final_rescale
)

def compress_model(
    pretrained_model: Module,
    finetuned_model: Module,
    patch_size: int,
    bit_strategy: List[Tuple[int, float]],
    transform_type: str = 'dct'  # New parameter to choose transform
) -> Dict[str, Any]:
    """
    Runs the complete compression pipeline on an entire model, using either DCT or DWT.

    Args:
        pretrained_model (torch.nn.Module): The original pre-trained model.
        finetuned_model (torch.nn.Module): The fine-tuned model.
        patch_size (int): The size of the patches for processing tensors.
        bit_strategy (List[Tuple[int, float]]): The mixed-precision allocation strategy.
        transform_type (str): The transform to use, either 'dct' or 'dwt'.

    Returns:
        Dict[str, Any]: A dictionary containing the compressed data for each layer.
    """
    print(f"Starting model compression using transform: {transform_type.upper()}...")
    
    delta_weights = calculate_delta_parameters(pretrained_model, finetuned_model)
    compressed_model_data = {}
    
    for layer_name, delta_tensor in delta_weights.items():
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
            
            # --- CHOOSE TRANSFORM BASED ON PARAMETER ---
            if transform_type == 'dwt':
                quantized_patches, min_vals, max_vals = dwt_and_quantize_patches(patches, bits)
            else: # Default to DCT
                quantized_patches, min_vals, max_vals = dct_and_quantize_patches(patches, bits)
            
            compressed_model_data[layer_name] = {
                'is_compressed': True,
                'transform_type': transform_type, # Store the transform type
                'quantized_patches': quantized_patches,
                'min_vals': min_vals,
                'max_vals': max_vals,
                'bit_allocations': bits,
                'original_shape': delta_tensor.shape,
                'patch_size': patch_size,
                'original_mean_abs': torch.mean(torch.abs(delta_tensor)).item()
            }
        else:
            print(f"Skipping compression for layer '{layer_name}'. Storing uncompressed.")
            compressed_model_data[layer_name] = {
                'is_compressed': False,
                'uncompressed_delta': delta_tensor
            }

    print("\nModel compression finished.")
    return compressed_model_data

def decompress_model(
    pretrained_model: Module,
    compressed_data: Dict[str, Any]
) -> Module:
    """
    Reconstructs a fine-tuned model from compressed delta data.
    """
    print("Starting model decompression...")
    reconstructed_model = copy.deepcopy(pretrained_model)
    
    for layer_name, layer_data in compressed_data.items():
        final_delta = None
        if layer_data['is_compressed']:
            print(f"Decompressing layer: {layer_name}...")
            transform_type = layer_data.get('transform_type', 'dct')
            
            # --- CHOOSE INVERSE TRANSFORM ---
            if transform_type == 'dwt':
                reconstructed_patches = dequantize_and_idwt_patches(
                    layer_data['quantized_patches'], layer_data['min_vals'],
                    layer_data['max_vals'], layer_data['bit_allocations'],
                    layer_data['patch_size']
                )
            else:
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
            param.data = (param.data.float() + final_delta.to(target_device))

    print("\nModel decompression finished.")
    return reconstructed_model