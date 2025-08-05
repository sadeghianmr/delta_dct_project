import pywt
import torch
from typing import List, Tuple
from scipy.fftpack import idctn

def dequantize_and_idct_patches(
    quantized_patches: List[torch.Tensor],
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    bit_allocations: torch.Tensor
) -> torch.Tensor:
    """
    Performs de-quantization and Inverse DCT on a list of compressed patches.
    """
    reconstructed_patches_list = []
    for i in range(len(quantized_patches)):
        quantized_patch, bits = quantized_patches[i].float(), bit_allocations[i].item()
        min_val, max_val = min_vals[i], max_vals[i]

        if bits > 0:
            if min_val == max_val:
                dequantized_dct = torch.full_like(quantized_patch, min_val)
            else:
                normalized = quantized_patch / (2**bits - 1)
                dequantized_dct = normalized * (max_val - min_val) + min_val
        else: # bits == 0
            dequantized_dct = torch.full_like(quantized_patch, min_val)
            
        reconstructed_patch = torch.from_numpy(idctn(dequantized_dct.numpy(), type=2, norm='ortho'))
        reconstructed_patches_list.append(reconstructed_patch)
        
    return torch.stack(reconstructed_patches_list)

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


def dequantize_and_idwt_patches(
    quantized_patches: List[torch.Tensor],
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    bit_allocations: torch.Tensor,
    original_patch_size: int
) -> torch.Tensor:
    """
    Performs de-quantization and Inverse DWT on a list of compressed patches.
    """
    reconstructed_patches_list = []
    # For IDWT, detail coefficients are needed. We'll approximate them with zeros.
    # The size of detail coefficients depends on the wavelet and original size.
    # For 'haar' DWT, the coefficient matrices are roughly half the size.
    coeff_shape = [s // 2 for s in (original_patch_size, original_patch_size)]
    LH = torch.zeros(coeff_shape)
    HL = torch.zeros(coeff_shape)
    HH = torch.zeros(coeff_shape)

    for i in range(len(quantized_patches)):
        quantized_patch, bits = quantized_patches[i].float(), bit_allocations[i].item()
        min_val, max_val = min_vals[i], max_vals[i]

        if bits > 0:
            if min_val == max_val:
                dequantized_ll = torch.full_like(quantized_patch, min_val)
            else:
                normalized = quantized_patch / (2**bits - 1)
                dequantized_ll = normalized * (max_val - min_val) + min_val
        else: # bits == 0
            dequantized_ll = torch.full_like(quantized_patch, min_val)

        # Reconstruct with zeroed detail coefficients
        coeffs = dequantized_ll.numpy(), (LH.numpy(), HL.numpy(), HH.numpy())
        reconstructed_patch = torch.from_numpy(pywt.idwt2(coeffs, 'haar'))
        
        # Resize to original patch size if necessary due to odd dimensions
        reconstructed_patch = torch.nn.functional.interpolate(
            reconstructed_patch.unsqueeze(0).unsqueeze(0), 
            size=(original_patch_size, original_patch_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).squeeze(0)

        reconstructed_patches_list.append(reconstructed_patch)
        
    return torch.stack(reconstructed_patches_list)