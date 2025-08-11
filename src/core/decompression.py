import pywt
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from scipy.fftpack import idctn

def patches_to_tensor(
    reconstructed_patches: torch.Tensor,
    original_shape: Tuple[int, int],
    patch_size: int
) -> torch.Tensor:
    """
    Reassembles a tensor of patches back into a single 2D tensor.
    """
    original_h, original_w = original_shape
    padded_h = ((original_h + patch_size - 1) // patch_size) * patch_size
    padded_w = ((original_w + patch_size - 1) // patch_size) * patch_size
    num_patches_h, num_patches_w = padded_h // patch_size, padded_w // patch_size
    patches_grid = reconstructed_patches.view(num_patches_h, num_patches_w, patch_size, patch_size)
    result = patches_grid.permute(0, 2, 1, 3).contiguous().view(padded_h, padded_w)
    return result[:original_h, :original_w]

def final_rescale(reconstructed_tensor: torch.Tensor, original_mean_abs: float) -> torch.Tensor:
    """
    Applies a final layer-wise rescaling using the original tensor's mean absolute value.
    """
    reconstructed_mean_abs = torch.mean(torch.abs(reconstructed_tensor))
    if reconstructed_mean_abs == 0:
        return reconstructed_tensor
    scale_factor = original_mean_abs / reconstructed_mean_abs
    return reconstructed_tensor * scale_factor


def dequantize_and_idct_patches(
    quantized_patches: List[torch.Tensor],
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    bit_allocations: torch.Tensor,
    q_table: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Performs de-quantization and Inverse DCT, with optional JPEG-style table."""
    reconstructed_patches_list = []
    for i in range(len(quantized_patches)):
        quantized_patch, bits = quantized_patches[i].float(), bit_allocations[i].item()
        min_val, max_val = min_vals[i], max_vals[i]
        dequantized_dct = torch.full_like(quantized_patch, 0.0)

        if bits > 0:
            if min_val == max_val:
                dequantized_dct = torch.full_like(quantized_patch, min_val)
            else:
                normalized = quantized_patch / (2**bits - 1)
                dequantized_dct = normalized * (max_val - min_val) + min_val
        else:
            dequantized_dct = torch.full_like(quantized_patch, min_val)
        
        # --- NEW: Apply JPEG quantization table if provided ---
        if q_table is not None:
            # print(f"DEBUG: Applying INVERSE JPEG Quantization in IDCT...") # DEBUG PRINT

            dequantized_dct = dequantized_dct * q_table

        reconstructed_patch = torch.from_numpy(idctn(dequantized_dct.numpy(), type=2, norm='ortho'))
        reconstructed_patches_list.append(reconstructed_patch)
        
    return torch.stack(reconstructed_patches_list)

def dequantize_and_idwt_patches(
    quantized_coeffs_list: List[List[torch.Tensor]],
    min_vals_tensor: torch.Tensor,
    max_vals_tensor: torch.Tensor,
    bit_allocations: torch.Tensor,
    original_patch_size: int,
    q_table: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Performs de-quantization and Inverse DWT. Reconstructs missing
    coefficients with zeros based on how many were provided.
    """
    reconstructed_patches_list = []

    for i in range(len(quantized_coeffs_list)):
        quantized_parts = quantized_coeffs_list[i]
        allocated_bits_for_ll = bit_allocations[i].item()
        min_vals = min_vals_tensor[i]
        max_vals = max_vals_tensor[i]
        
        bits_for_parts = [allocated_bits_for_ll, 1, 1, 1]

        dequantized_coeffs = []
        for j in range(len(quantized_parts)): # Loop over the provided quantized parts
            quantized_patch = quantized_parts[j].float()
            current_bits = bits_for_parts[j]
            min_val, max_val = min_vals[j], max_vals[j]
            dequantized_part = torch.full_like(quantized_patch, 0.0)

            if current_bits > 0:
                if min_val == max_val:
                    dequantized_part = torch.full_like(quantized_patch, min_val)
                else:
                    normalized = quantized_patch / (2**current_bits - 1)
                    dequantized_part = normalized * (max_val - min_val) + min_val
            else:
                dequantized_part = torch.full_like(quantized_patch, min_val)

            if q_table is not None:
                coeff_size = dequantized_part.shape[0]
                if q_table.shape[0] != coeff_size:
                    resized_q_table = F.interpolate(q_table.view(1, 1, *q_table.shape), size=(coeff_size, coeff_size), mode='bilinear', align_corners=False).squeeze().clamp(min=1.0)
                else:
                    resized_q_table = q_table
                dequantized_part = dequantized_part * resized_q_table

            dequantized_coeffs.append(dequantized_part.numpy())

        # --- NEW LOGIC: Reconstruct based on how many coeffs were saved ---
        # Default to zeros for any missing coefficients
        num_coeffs_saved = len(dequantized_coeffs)
        coeff_names = ['LL', 'LH', 'HL', 'HH']
        final_coeffs = {}

        # Fill in the saved coeffs
        for j in range(num_coeffs_saved):
            final_coeffs[coeff_names[j]] = dequantized_coeffs[j]
        
        # Fill in the missing coeffs with zeros
        coeff_shape = dequantized_coeffs[0].shape # Get shape from LL
        for j in range(num_coeffs_saved, 4):
            final_coeffs[coeff_names[j]] = torch.zeros(coeff_shape).numpy()

        coeffs = final_coeffs['LL'], (final_coeffs['LH'], final_coeffs['HL'], final_coeffs['HH'])
        reconstructed_patch = torch.from_numpy(pywt.idwt2(coeffs, 'haar'))
        
        reconstructed_patch = F.interpolate(
            reconstructed_patch.unsqueeze(0).unsqueeze(0),
            size=(original_patch_size, original_patch_size),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        reconstructed_patches_list.append(reconstructed_patch)
        
    return torch.stack(reconstructed_patches_list)