"""
Pure PyTorch replacement for flex_gemm.ops.grid_sample.grid_sample_3d.

flex_gemm is CUDA-only. This module reimplements grid_sample_3d by scattering
sparse attributes into a dense volume and using torch.nn.functional.grid_sample.

On Strix Halo with unified memory, the extra dense volume allocation is acceptable
(CPU and GPU share the same memory pool).
"""

import torch
import torch.nn.functional as F


def grid_sample_3d(attr_volume, coords, shape, grid, mode='trilinear'):
    """Sample a sparse 3D attribute volume at arbitrary continuous positions.

    Drop-in replacement for flex_gemm.ops.grid_sample.grid_sample_3d.

    Args:
        attr_volume: [N, C] attribute tensor (sparse voxel attributes)
        coords: [N, 4] integer coordinates with batch index prepended (batch, x, y, z)
        shape: torch.Size([B, C, D, H, W]) target dense volume shape
        grid: [B, M, 3] query positions in voxel space (not normalized)
        mode: interpolation mode ('trilinear' mapped to 'bilinear' for grid_sample)

    Returns:
        [M, C] sampled attributes (squeezed from batch dim for B=1)
        or [B, M, C] if B > 1
    """
    B, C = shape[0], shape[1]
    D, H, W = shape[2], shape[3], shape[4]
    device = attr_volume.device
    dtype = attr_volume.dtype

    # Build dense volume from sparse data
    dense = torch.zeros(B, C, D, H, W, device=device, dtype=dtype)

    batch_idx = coords[:, 0].long()
    x = coords[:, 1].long()
    y = coords[:, 2].long()
    z = coords[:, 3].long()

    # Clamp to valid range
    x = x.clamp(0, D - 1)
    y = y.clamp(0, H - 1)
    z = z.clamp(0, W - 1)

    # Scatter attributes into dense volume
    dense[batch_idx, :, x, y, z] = attr_volume.to(dtype)

    # Normalize grid to [-1, 1] for torch.nn.functional.grid_sample
    grid_norm = torch.empty_like(grid)
    grid_norm[..., 0] = 2.0 * grid[..., 0] / (D - 1) - 1.0 if D > 1 else 0.0
    grid_norm[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0 if H > 1 else 0.0
    grid_norm[..., 2] = 2.0 * grid[..., 2] / (W - 1) - 1.0 if W > 1 else 0.0

    # Clamp to [-1, 1] to avoid out-of-bounds
    grid_norm = grid_norm.clamp(-1.0, 1.0)

    # grid_sample expects [B, D_out, H_out, W_out, 3]
    M = grid.shape[1]
    grid_5d = grid_norm.reshape(B, 1, 1, M, 3)

    # Map mode names
    gs_mode = 'bilinear' if mode == 'trilinear' else mode

    result = F.grid_sample(
        dense, grid_5d,
        mode=gs_mode,
        padding_mode='zeros',
        align_corners=True,
    )

    # result shape: [B, C, 1, 1, M]
    result = result.reshape(B, C, M).permute(0, 2, 1)  # [B, M, C]

    # Match original API: returns [M, C] when B=1
    if B == 1:
        return result[0]
    return result
