"""Compatibility aliases for the former HRQ name.

New encoded states use the versioned ``trq`` format. Existing identity HRQ
states remain decodable; legacy affine states require explicit parameters.
"""

from .trq import (
    extract_trq_bits,
    trq_dequantize_tensor,
    trq_quantize_tensor,
    trq_state_nbytes,
)


extract_hrq_bits = extract_trq_bits
hrq_quantize_tensor = trq_quantize_tensor
hrq_dequantize_tensor = trq_dequantize_tensor


__all__ = [
    "extract_hrq_bits",
    "hrq_quantize_tensor",
    "hrq_dequantize_tensor",
    "trq_state_nbytes",
]
