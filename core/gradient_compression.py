"""
gradient_compression.py
New Master's contribution — compression layer for client→shard-chief transmission.
Implements Option A with residual/error-feedback accumulation per Chapter 3.
"""

import numpy as np
import pickle
from typing import Tuple, Dict, Any


def compress_update_with_residual(gradient: np.ndarray,
                                   residual: np.ndarray,
                                   method: str = "topk",
                                   ratio: float = 0.1,
                                   quantize_bits: int = 8) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """
    Compress a gradient update with residual/error-feedback compensation (Procedure 1, Chapter 3).

    Algorithm:
    1. Compensated update: Δ'Wi = ΔWi + ri
    2. Compress Δ'Wi via compress_update_kernel
    3. Residual for next round: ri := Δ'Wi, then zero the retained coordinates

    Args:
        gradient: raw update ΔWi (already clipped)
        residual: accumulated residual ri from previous round
        method: "topk" | "quantize" | "topk_quantize"
        ratio: fraction of values to keep (for topk)
        quantize_bits: bit-width for quantization

    Returns:
        (compressed_payload, meta, new_residual) where:
        - compressed_payload and meta allow shard-side decompression
        - new_residual is the updated residual for the next round
    """
    # Step 1: Error compensation
    compensated = gradient + residual
    
    # Step 2: Compress the compensated update
    payload, meta = compress_update_kernel(compensated, method, ratio, quantize_bits)
    
    # Step 3: Update residual (zero out retained coordinates)
    if method in ("topk", "topk_quantize"):
        indices = meta.get("indices", np.array([], dtype=np.int32))
        new_residual = compensated.copy().flatten()
        new_residual[indices] = 0.0
        new_residual = new_residual.reshape(compensated.shape)
    else:
        new_residual = np.zeros_like(compensated)
    
    return payload, meta, new_residual


def compress_update_kernel(gradient: np.ndarray,
                            method: str = "topk",
                            ratio: float = 0.1,
                            quantize_bits: int = 8) -> Tuple[Any, Dict[str, Any]]:
    """
    Core compression kernel (used by both compress_update_with_residual and standalone compression).
    Compress a gradient update before transmission.

    Args:
        gradient: full-precision, full-dimension update (with or without residual compensation)
        method: "topk" | "quantize" | "topk_quantize"
        ratio: fraction of values to keep (for topk / topk_quantize)
        quantize_bits: bit-width for quantization (for quantize / topk_quantize)

    Returns:
        (compressed_payload, meta) where meta contains everything needed
        to reconstruct a full-size array (original shape, method, indices
        if sparsified, min/max if quantized, dtype).
    """
    original_shape = gradient.shape
    flat = gradient.flatten()

    meta: Dict[str, Any] = {
        "method": method,
        "original_shape": original_shape,
        "original_dtype": str(gradient.dtype),
        "original_num_params": flat.size,
    }

    if method in ("topk", "topk_quantize"):
        k = max(1, int(flat.size * ratio))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        values = flat[idx]
        meta["indices"] = idx.astype(np.int32)
        meta["kept_count"] = k
        payload_values = values
    else:
        payload_values = flat

    if method in ("quantize", "topk_quantize"):
        vmin, vmax = float(payload_values.min()), float(payload_values.max())
        scale = (vmax - vmin) if vmax > vmin else 1.0
        q = np.round((payload_values - vmin) / scale * (2 ** quantize_bits - 1))
        q = q.astype(np.uint8 if quantize_bits <= 8 else np.uint16)
        meta["quantize_bits"] = quantize_bits
        meta["vmin"], meta["vmax"] = vmin, vmax
        payload_values = q

    return payload_values, meta


# Backwards-compatibility alias for standalone (no residual) compression
def compress_update(gradient: np.ndarray,
                     method: str = "topk",
                     ratio: float = 0.1,
                     quantize_bits: int = 8) -> Tuple[Any, Dict[str, Any]]:
    """
    Standalone compression without residual (for compatibility).
    Use compress_update_with_residual for error-feedback enabled path.
    """
    return compress_update_kernel(gradient, method, ratio, quantize_bits)


def decompress_update(payload_values: Any, meta: Dict[str, Any]) -> np.ndarray:
    """
    Reconstruct a full dense array from a compressed payload.
    Downstream code (shard_chief.py) never needs to know compression happened.
    """
    if not meta or meta.get("method") == "identity":
        return np.asarray(payload_values)

    method = meta["method"]
    values = payload_values

    if method in ("quantize", "topk_quantize"):
        scale = (meta["vmax"] - meta["vmin"]) if meta["vmax"] > meta["vmin"] else 1.0
        values = values.astype(np.float32) / (2 ** meta["quantize_bits"] - 1) * scale + meta["vmin"]

    full = np.zeros(meta["original_num_params"], dtype=np.float32)
    if method in ("topk", "topk_quantize"):
        full[meta["indices"]] = values
    else:
        full[:] = values

    return full.reshape(meta["original_shape"])


def measure_serialized_size(obj: Any) -> int:
    """
    Return the REAL number of bytes this object would occupy on the wire,
    by actually serializing it (even though transmission is still in-process).
    This gives defensible, non-estimated numbers for the evaluation chapter.
    """
    return len(pickle.dumps(obj))
