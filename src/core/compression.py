import pywt
import torch
import torch.nn.functional as F
from typing import List, Tuple
from scipy.fftpack import dctn
from typing import List, Tuple, Optional, Any

def tensor_to_patches(delta_tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Divides a 2D tensor into smaller square patches.

    If the tensor's dimensions are not perfectly divisible by the patch_size,
    the tensor is padded with zeros to ensure complete coverage.

    Args:
        delta_tensor (torch.Tensor): The input 2D tensor, e.g., a layer's delta weights.
        patch_size (int): The edge size of the square patches (e.g., 8, 16).

    Returns:
        torch.Tensor: A 3D tensor of patches with the shape (num_patches, patch_size, patch_size).
    """
    if delta_tensor.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional.")
    height, width = delta_tensor.shape
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    padded_tensor = F.pad(delta_tensor, (0, pad_w, 0, pad_h))
    patches = padded_tensor.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size, patch_size)
    return patches

def calculate_importance_scores(patches_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the importance score for each patch using its L2 Norm.

    Args:
        patches_tensor (torch.Tensor): A 3D tensor of patches with the shape 
                                       (num_patches, patch_size, patch_size).

    Returns:
        torch.Tensor: A 1D tensor containing the L2 norm score for each patch.
    """
    if patches_tensor.dim() != 3:
        raise ValueError("Input must be a 3D tensor of patches.")
    num_patches = patches_tensor.shape[0]
    flattened_patches = patches_tensor.view(num_patches, -1)
    scores = torch.linalg.norm(flattened_patches, ord=2, dim=1)
    return scores

def calculate_post_transform_scores(transformed_patches: torch.Tensor, energy_size: int = 4) -> torch.Tensor:
    """
    Calculates importance score based on the energy in the top-left (low-frequency)
    corner of a transformed patch (post-transform).

    Args:
        transformed_patches (torch.Tensor): A 3D tensor of patches that have already
                                            undergone DCT or DWT.
        energy_size (int): The size of the top-left square to consider for energy calculation.

    Returns:
        torch.Tensor: A 1D tensor of importance scores.
    """
    if transformed_patches.dim() != 3:
        raise ValueError("Input must be a 3D tensor of transformed patches.")
    
    # Take the top-left corner of each patch (e.g., a 4x4 sub-matrix)
    low_freq_coeffs = transformed_patches[:, :energy_size, :energy_size]
    
    # Flatten this corner for each patch
    flattened_coeffs = low_freq_coeffs.contiguous().view(low_freq_coeffs.shape[0], -1)
    
    # Calculate the L2 norm (energy) of this corner
    scores = torch.linalg.norm(flattened_coeffs, ord=2, dim=1)
    return scores


def allocate_bit_widths(scores: torch.Tensor, allocation_map: List[Tuple[int, float]]) -> torch.Tensor:
    """
    Allocates quantization bit-widths to patches based on their importance scores.

    Args:
        scores (torch.Tensor): A 1D tensor of importance scores for each patch.
        allocation_map (List[Tuple[int, float]]): A list where each tuple contains
                                                  (bit_width, ratio). Ratios must sum to 1.0.

    Returns:
        torch.Tensor: A 1D integer tensor where each element is the allocated
                      bit-width for the corresponding patch.
    """
    total_ratio = sum(ratio for _, ratio in allocation_map)
    if not torch.isclose(torch.tensor(total_ratio), torch.tensor(1.0)):
        raise ValueError("The sum of ratios in allocation_map must be 1.0.")
    num_patches = scores.shape[0]
    sorted_indices = torch.argsort(scores, descending=True)
    bit_allocations = torch.full_like(scores, -1, dtype=torch.int)
    current_pos = 0
    for bit_width, ratio in allocation_map:
        num_for_this_bit = round(num_patches * ratio)
        end_pos = min(current_pos + num_for_this_bit, num_patches)
        indices_to_assign = sorted_indices[current_pos:end_pos]
        bit_allocations[indices_to_assign] = bit_width
        current_pos = end_pos
    return bit_allocations

def dct_and_quantize_patches(patches_tensor: torch.Tensor, bit_allocations: torch.Tensor,
                             q_table: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Applies DCT and quantization, with optional JPEG-style quantization."""
    num_patches = patches_tensor.shape[0]
    quantized_patches_list = []
    min_vals = torch.zeros(num_patches, dtype=torch.float32)
    max_vals = torch.zeros(num_patches, dtype=torch.float32)
    patches_tensor = patches_tensor.float()

    for i in range(num_patches):
        patch = patches_tensor[i]
        bits = bit_allocations[i].item()
        dct_patch = torch.from_numpy(dctn(patch.numpy(), type=2, norm='ortho')).float()
        
        # --- NEW: Apply JPEG quantization table if provided ---
        if q_table is not None:
            # print(f"DEBUG: Applying JPEG Quantization Table in DCT...") # DEBUG PRINT
            dct_patch = dct_patch / q_table

        if bits > 0:
            min_val, max_val = dct_patch.min(), dct_patch.max()
            min_vals[i], max_vals[i] = min_val, max_val
            if min_val == max_val:
                quantized = torch.zeros_like(dct_patch, dtype=torch.int8)
            else:
                normalized = (dct_patch - min_val) / (max_val - min_val)
                scaled = normalized * (2**bits - 1)
                quantized = torch.round(scaled).to(torch.int8)
            quantized_patches_list.append(quantized)
        else:
            avg_val = patch.mean()
            min_vals[i], max_vals[i] = avg_val, avg_val
            quantized_patches_list.append(torch.zeros_like(patch, dtype=torch.int8))

    return quantized_patches_list, min_vals, max_vals


def dwt_and_quantize_patches(
    patches_tensor: torch.Tensor,
    bit_allocations: torch.Tensor,
    q_table: Optional[torch.Tensor] = None,
    coeffs_to_keep: str = 'all'  # NEW: Parameter to control which coeffs to save
) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
    """
    Applies DWT and quantizes a selected set of coefficients based on the coeffs_to_keep strategy.
    """
    num_patches = patches_tensor.shape[0]
    quantized_coeffs_list = []
    min_vals_list = []
    max_vals_list = []
    
    patches_tensor = patches_tensor.float()

    for i in range(num_patches):
        patch = patches_tensor[i]
        allocated_bits_for_ll = bit_allocations[i].item()

        coeffs = pywt.dwt2(patch.numpy(), 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # --- NEW LOGIC: Select which coefficients to process ---
        all_coeffs_map = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}
        
        if coeffs_to_keep == 'll_only':
            coeffs_map_to_process = {'LL': LL}
        elif coeffs_to_keep == 'll_lh_hl':
            coeffs_map_to_process = {'LL': LL, 'LH': LH, 'HL': HL}
        else:  # Default to 'all'
            coeffs_map_to_process = all_coeffs_map

        dwt_patches = [torch.from_numpy(c).float() for c in coeffs_map_to_process.values()]
        
        bits_for_parts = [allocated_bits_for_ll, 1, 1, 1]

        quantized_parts = []
        min_vals_parts = []
        max_vals_parts = []

        # This loop now iterates over only the selected coefficients
        for j, coeff_patch in enumerate(dwt_patches):
            current_bits = bits_for_parts[j]
            temp_patch = coeff_patch
            
            if q_table is not None:
                coeff_size = temp_patch.shape[0]
                if q_table.shape[0] != coeff_size:
                    resized_q_table = F.interpolate(q_table.view(1, 1, *q_table.shape), size=(coeff_size, coeff_size), mode='bilinear', align_corners=False).squeeze().clamp(min=1.0)
                else:
                    resized_q_table = q_table
                temp_patch = temp_patch / resized_q_table

            if current_bits > 0:
                min_val, max_val = temp_patch.min(), temp_patch.max()
                if min_val == max_val:
                    quantized = torch.zeros_like(temp_patch, dtype=torch.int8)
                else:
                    normalized = (temp_patch - min_val) / (max_val - min_val)
                    scaled = normalized * (2**current_bits - 1)
                    quantized = torch.round(scaled).to(torch.int8)
                
                quantized_parts.append(quantized)
                min_vals_parts.append(min_val)
                max_vals_parts.append(max_val)
            else: # bits == 0
                avg_val = temp_patch.mean()
                quantized_parts.append(torch.zeros_like(temp_patch, dtype=torch.int8))
                min_vals_parts.append(avg_val)
                max_vals_parts.append(avg_val)

        quantized_coeffs_list.append(quantized_parts)
        min_vals_list.append(torch.tensor(min_vals_parts))
        max_vals_list.append(torch.tensor(max_vals_parts))

    return quantized_coeffs_list, torch.stack(min_vals_list), torch.stack(max_vals_list)