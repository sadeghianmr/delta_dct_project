import pywt
import torch
import torch.nn.functional as F
from typing import List, Tuple
from scipy.fftpack import dctn
from typing import List, Tuple, Optional

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

def dwt_and_quantize_patches(patches_tensor: torch.Tensor, bit_allocations: torch.Tensor,
                             q_table: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Applies DWT and quantization, with optional JPEG-style quantization."""
    num_patches = patches_tensor.shape[0]
    quantized_patches_list = []
    min_vals = torch.zeros(num_patches, dtype=torch.float32)
    max_vals = torch.zeros(num_patches, dtype=torch.float32)
    patches_tensor = patches_tensor.float()

    for i in range(num_patches):
        patch = patches_tensor[i]
        bits = bit_allocations[i].item()
        coeffs = pywt.dwt2(patch.numpy(), 'haar')
        LL, (LH, HL, HH) = coeffs
        dwt_patch = torch.from_numpy(LL).float()

        # --- NEW: Apply JPEG quantization table if provided ---
        if q_table is not None:
            # print(f"DEBUG: Applying JPEG Quantization Table in DWT...") # DEBUG PRINT
            dwt_patch_size = dwt_patch.shape[0]
            if q_table.shape[0] != dwt_patch_size:
                resized_q_table = F.interpolate(q_table.view(1, 1, *q_table.shape), size=(dwt_patch_size, dwt_patch_size), mode='bilinear').squeeze().clamp(min=1.0)
            else:
                resized_q_table = q_table
            dwt_patch = dwt_patch / resized_q_table

        if bits > 0:
            min_val, max_val = dwt_patch.min(), dwt_patch.max()
            min_vals[i], max_vals[i] = min_val, max_val
            if min_val == max_val:
                quantized = torch.zeros_like(dwt_patch, dtype=torch.int8)
            else:
                normalized = (dwt_patch - min_val) / (max_val - min_val)
                scaled = normalized * (2**bits - 1)
                quantized = torch.round(scaled).to(torch.int8)
            quantized_patches_list.append(quantized)
        else:
            avg_val = patch.mean()
            min_vals[i], max_vals[i] = avg_val, avg_val
            quantized_patches_list.append(torch.zeros_like(dwt_patch, dtype=torch.int8))

    return quantized_patches_list, min_vals, max_vals