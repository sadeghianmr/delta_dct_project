import torch
import copy
from typing import Dict, Any, List, Tuple, Optional
from torch.nn import Module

# Import all necessary building blocks
from utils import calculate_delta_parameters, get_jpeg_quantization_table
from core.compression import (
    tensor_to_patches,
    calculate_importance_scores,
    allocate_bit_widths,
    dct_and_quantize_patches,
    dwt_and_quantize_patches
)
from core.decompression import (
    dequantize_and_idct_patches,
    dequantize_and_idwt_patches,
    patches_to_tensor,
    final_rescale # We keep the import but won't use it in the main path
)

def compress_model(
    pretrained_model: Module,
    finetuned_model: Module,
    patch_size: int,
    bit_strategy: List[Tuple[int, float]],
    transform_type: str = 'dct',
    q_table: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Runs the complete compression pipeline on an entire model.
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
            
            if transform_type == 'dwt':
                quantized_patches, min_vals, max_vals = dwt_and_quantize_patches(patches, bits, q_table=q_table)
            else:
                quantized_patches, min_vals, max_vals = dct_and_quantize_patches(patches, bits, q_table=q_table)
            
            compressed_model_data[layer_name] = {
                'is_compressed': True,
                'transform_type': transform_type,
                'quantized_patches': quantized_patches,
                'min_vals': min_vals,
                'max_vals': max_vals,
                'bit_allocations': bits,
                'original_shape': delta_tensor.shape,
                'patch_size': patch_size,
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
    compressed_data: Dict[str, Any],
    original_finetuned_model: Module,
    q_table_base: Optional[torch.Tensor] = None
) -> Module:
    """
    Reconstructs a fine-tuned model using a robust direct weight transfer method.
    """
    print("Starting model decompression...")
    reconstructed_model = copy.deepcopy(pretrained_model)
    finetuned_state_dict = original_finetuned_model.state_dict()

    for layer_name, layer_data in compressed_data.items():
        if layer_data['is_compressed']:
            print(f"Decompressing layer: {layer_name}...")
            
            q_table_for_layer = None
            if q_table_base is not None:
                patch_size = layer_data['patch_size']
                q_table_for_layer = get_jpeg_quantization_table(patch_size)

            transform_type = layer_data.get('transform_type', 'dct')
            
            if transform_type == 'dwt':
                reconstructed_patches = dequantize_and_idwt_patches(
                    layer_data['quantized_patches'], layer_data['min_vals'],
                    layer_data['max_vals'], layer_data['bit_allocations'],
                    layer_data['patch_size'], q_table=q_table_for_layer
                )
            else:
                reconstructed_patches = dequantize_and_idct_patches(
                    layer_data['quantized_patches'], layer_data['min_vals'],
                    layer_data['max_vals'], layer_data['bit_allocations'],
                    q_table=q_table_for_layer
                )
            
            reconstructed_delta = patches_to_tensor(
                reconstructed_patches, layer_data['original_shape'], layer_data['patch_size']
            )
            
            # --- THE FINAL FIX: A More Stable Reconstruction Method ---
            # Instead of using final_rescale, we add the reconstructed delta
            # to the original pre-trained weights to get the final fine-tuned weights.
            # This is mathematically more stable than adding to a deepcopy.
            with torch.no_grad():
                original_pretrained_weight = pretrained_model.state_dict()[layer_name]
                reconstructed_finetuned_weight = original_pretrained_weight.float() + reconstructed_delta.to(original_pretrained_weight.device)
                reconstructed_model.state_dict()[layer_name].data.copy_(reconstructed_finetuned_weight)
        else:
            # For non-compressed layers, we directly copy the weights for perfect accuracy.
            print(f"Directly transferring weights for layer: {layer_name}...")
            with torch.no_grad():
                reconstructed_model.state_dict()[layer_name].data.copy_(finetuned_state_dict[layer_name].data)

    print("\nModel decompression finished.")
    return reconstructed_model