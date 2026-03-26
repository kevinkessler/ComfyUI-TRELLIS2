#!/bin/bash
# =============================================================================
# ROCm 7.2 setup for ComfyUI-TRELLIS2 on AMD Strix Halo APU
# =============================================================================
#
# This script sets up the environment for running TRELLIS.2 on AMD Strix Halo
# APUs (RDNA 3.5, gfx1151) using ROCm 7.2 instead of NVIDIA CUDA.
#
# Usage: source setup_rocm.sh
#        (or: bash setup_rocm.sh --install to also install packages)

set -euo pipefail

# ---- Strix Halo GPU identification ----
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1151

# ---- OpenGL/EGL for nvdiffrast (headless rasterization) ----
export PYOPENGL_PLATFORM=egl

# ---- Verify ROCm is working ----
echo "=== ROCm Environment Setup for Strix Halo ==="
echo ""

if command -v rocminfo &> /dev/null; then
    echo "ROCm installation found:"
    rocminfo 2>/dev/null | grep -E "Name:|Marketing Name:" | head -4
    echo ""
else
    echo "WARNING: rocminfo not found. Is ROCm 7.2 installed?"
    echo "Install: https://rocm.docs.amd.com/projects/install-on-linux/"
    echo ""
fi

# ---- Package installation (only with --install flag) ----
if [[ "${1:-}" == "--install" ]]; then
    echo "=== Installing packages ==="

    # 1. PyTorch for ROCm 7.2
    echo ""
    echo "--- Installing PyTorch (ROCm 7.2) ---"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2

    # 2. nvdiffrast (OpenGL backend — no CUDA needed)
    echo ""
    echo "--- Installing nvdiffrast (OpenGL backend) ---"
    pip install nvdiffrast

    # 3. CPU mesh processing dependencies
    echo ""
    echo "--- Installing CPU mesh processing deps ---"
    pip install xatlas pyfqmr

    # 4. EGL dependencies (for headless nvdiffrast)
    echo ""
    echo "--- Checking EGL libraries ---"
    if ! ldconfig -p 2>/dev/null | grep -q libEGL; then
        echo "WARNING: libEGL not found. Install with:"
        echo "  sudo apt install libegl1-mesa-dev"
    else
        echo "libEGL found."
    fi

    # 5. Remaining Python deps
    echo ""
    echo "--- Installing remaining Python dependencies ---"
    pip install -r requirements.txt

    # 6. Optional: flash-attention with ROCm triton backend
    echo ""
    echo "--- Flash Attention (optional) ---"
    echo "To install flash-attention for ROCm (optional, SDPA fallback is automatic):"
    echo "  git clone https://github.com/ROCm/flash-attention.git"
    echo "  cd flash-attention"
    echo "  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install --no-build-isolation ."
    echo ""
    echo "Note: flash-attn on RDNA 3.5 may only work with the triton backend."
    echo "If it fails, the system will fall back to PyTorch SDPA automatically."

    echo ""
    echo "=== Installation complete ==="
fi

# ---- Verify PyTorch + ROCm ----
echo ""
echo "--- Verifying PyTorch ROCm ---"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Device count: {torch.cuda.device_count()}')
else:
    print('WARNING: No GPU detected. Check ROCm installation and HSA_OVERRIDE_GFX_VERSION.')
" 2>/dev/null || echo "WARNING: PyTorch not installed yet. Run: source setup_rocm.sh --install"

echo ""
echo "Environment variables set:"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
echo "  PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
echo ""
echo "Ready. Start ComfyUI as usual."
