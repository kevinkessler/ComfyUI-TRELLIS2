# ComfyUI-TRELLIS2 — ROCm Port

Fork of [PozzettiAndrea/ComfyUI-TRELLIS2](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2) with support for AMD GPUs via ROCm 7.2+.

Tested on AMD Strix Halo APU (RDNA 3.5, gfx1151) with unified memory.

## What Changed

The upstream TRELLIS2 node depends on several CUDA-only libraries (`cumesh`, `flex_gemm`, `o_voxel`, `spconv`, `sageattention`). This fork replaces them with pure PyTorch and CPU fallbacks so the full pipeline runs on ROCm without any CUDA-only binaries.

### CUDA-Only Replacements

| Original (CUDA-only) | Replacement | File |
|---|---|---|
| `cumesh` (GPU mesh ops) | CPU mesh processing via `trimesh`, `xatlas`, `pyfqmr` | `nodes/rocm_mesh_ops.py` |
| `flex_gemm.ops.grid_sample.grid_sample_3d` | Dense scatter + `torch.nn.functional.grid_sample` | `nodes/rocm_grid_sample.py` |
| `o_voxel` (voxel-to-mesh) | Pure PyTorch dual contouring | `nodes/rocm_voxel_ops.py` |
| `spconv` / `torchsparse` / `flex_gemm` (sparse conv) | Pure PyTorch gather + matmul sparse convolution | `ops_sparse_patched.py` (see [ComfyUI Core Patch](#comfyui-core-patch)) |
| `sageattention` | Removed (PyTorch SDPA used instead) | `comfy-env-root.toml` |
| `nvdiffrast` CUDA context | `nvdiffrast` OpenGL/EGL context | `nodes/nodes_unwrap.py` |

### Modified Files (from upstream)

- **`nodes/trellis2/vae.py`** — Replaced `cumesh`, `o_voxel`, and `flex_gemm` imports with ROCm-compatible fallbacks.
- **`nodes/nodes_unwrap.py`** — Replaced `cumesh` with `rocm_mesh_ops`, `flex_gemm.grid_sample` with `rocm_grid_sample`, and switched nvdiffrast from `RasterizeCudaContext` to `RasterizeGLContext`.
- **`nodes/stages.py`** — Replaced `cumesh` import with `rocm_mesh_ops`.
- **`requirements.txt`** — Added `xatlas`, `pyfqmr`, `nvdiffrast` as direct dependencies.
- **`comfy-env-root.toml`** / **`nodes/comfy-env.toml`** — Removed CUDA-only package declarations.

### New Files

- **`nodes/rocm_mesh_ops.py`** — Drop-in `CuMeshCompat` class replacing `cumesh.CuMesh` using trimesh, xatlas, and pyfqmr for CPU-based mesh simplification, UV unwrapping, and hole filling.
- **`nodes/rocm_grid_sample.py`** — Drop-in `grid_sample_3d` replacing `flex_gemm.ops.grid_sample.grid_sample_3d` using dense volume scatter + `F.grid_sample`.
- **`nodes/rocm_voxel_ops.py`** — Drop-in `flexible_dual_grid_to_mesh` and `to_glb` replacing `o_voxel` operations with pure PyTorch dual contouring.
- **`setup_rocm.sh`** — Environment setup and package installation script for ROCm.

## ComfyUI Core Patch

The sparse convolution backend lives in ComfyUI core, not in this custom node. The file `ops_sparse_patched.py` in this repo is a patched version of `comfy/ops_sparse.py` that adds a pure PyTorch fallback backend.

**You must install this patch manually** — copy it into your ComfyUI installation:

```bash
cp ops_sparse_patched.py /path/to/ComfyUI/comfy/ops_sparse.py
```

### What the patch does

When no native sparse conv library (spconv, torchsparse, flex_gemm) is detected, the original code crashes with `KeyError: 'none'`. The patch adds a `'pytorch'` backend that:

1. **Builds a neighbor index map** — hashes voxel coordinates and uses binary search to find each voxel's 3x3x3 neighborhood. The map is cached in the SparseTensor's `spatial_cache` so it's computed only once per unique spatial layout.
2. **Performs submanifold sparse convolution** — loops over 27 kernel positions, gathers neighbor features via the cached index map, and accumulates via `addmm_` (one matrix multiply per kernel position).
3. **Weight format** — uses `[C_out, Kd, Kh, Kw, C_in]` layout, matching the TRELLIS2 checkpoint format.

The fallback only supports submanifold convolution (stride=1, no padding), which is all TRELLIS2 uses. Inverse/transposed sparse convolution is not implemented.

## Installation

### Prerequisites

- ROCm 7.2+ installed and working (`rocminfo` shows your GPU)
- ComfyUI installed in a venv with PyTorch for ROCm
- OpenGL/EGL libraries for nvdiffrast (`sudo apt install libegl1-mesa-dev`)

### Quick Setup

```bash
# 1. Clone this repo into ComfyUI custom_nodes
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/kevinkessler/ComfyUI-TRELLIS2.git

# 2. Set environment variables (add to your shell profile or container entrypoint)
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # Adjust for your GPU
export PYTORCH_ROCM_ARCH=gfx1151          # Adjust for your GPU
export PYOPENGL_PLATFORM=egl

# 3. Install dependencies
pip install -r ComfyUI-TRELLIS2/requirements.txt
pip install xatlas pyfqmr nvdiffrast

# 4. Apply the ComfyUI core patch for sparse convolution
cp ComfyUI-TRELLIS2/ops_sparse_patched.py /path/to/ComfyUI/comfy/ops_sparse.py

# 5. Start ComfyUI as usual
```

Or use the provided setup script:

```bash
source ComfyUI-TRELLIS2/setup_rocm.sh --install
```

### GPU-Specific Notes

The `HSA_OVERRIDE_GFX_VERSION` and `PYTORCH_ROCM_ARCH` values depend on your AMD GPU:

| GPU Family | `HSA_OVERRIDE_GFX_VERSION` | `PYTORCH_ROCM_ARCH` |
|---|---|---|
| Strix Halo (RDNA 3.5) | `11.0.0` | `gfx1151` |
| RX 7900 XTX (RDNA 3) | `11.0.0` | `gfx1100` |
| RX 7600 (RDNA 3) | `11.0.0` | `gfx1102` |

Consult the [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) for other GPUs.

## Known Limitations

- **Performance**: The pure PyTorch sparse convolution is slower than native spconv/flex_gemm. Shape decode takes longer but produces correct results.
- **Memory**: The gather-based convolution and dense grid_sample use more memory than optimized sparse implementations. On unified memory systems (Strix Halo APU) this is acceptable.
- **ComfyUI-GeometryPack**: The `cumesh`-based GPU remeshing node in GeometryPack still requires CUDA. Bypass it in your workflow or use CPU-based remeshing alternatives.
- **Inverse sparse conv**: Not implemented in the PyTorch fallback. TRELLIS2 does not use it, but other models might.

## Upstream

Based on [PozzettiAndrea/ComfyUI-TRELLIS2](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2) v0.2.3 (commit `b3a87ad`).

TRELLIS.2 model by Microsoft Research: [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS).
