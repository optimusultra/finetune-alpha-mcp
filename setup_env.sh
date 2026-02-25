#!/bin/bash
# Alpha-MCP FunctionGemma — uv Environment Bootstrap
# =====================================================
# Run this once to create the isolated .venv and install all dependencies.
# CUDA 12.1 build of PyTorch is used by default (GTX 1060 compatible via fp16).

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════╗"
echo "║   Alpha-MCP Fine-tuning — uv Environment Setup      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Create virtual environment ─────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "⏳ Creating .venv with Python 3.10..."
    uv venv .venv --python 3.10
    echo "✅ .venv created"
else
    echo "✅ .venv already exists — skipping creation"
fi

# ── Install PyTorch with CUDA 12.1 ────────────────────────────────────────────
echo ""
echo "⏳ Installing PyTorch (CUDA 12.1)..."
uv pip install \
    --python .venv \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# ── Install ML training stack ─────────────────────────────────────────────────
echo ""
echo "⏳ Installing training stack (transformers, trl, peft, datasets)..."
uv pip install \
    --python .venv \
    transformers>=4.40.0 \
    trl>=0.8.0 \
    peft>=0.10.0 \
    datasets>=2.18.0 \
    huggingface-hub>=0.22.0 \
    accelerate>=0.27.0 \
    ipykernel

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "═══════════════════════════════════════════"
echo " ACTIVATE + RUN:"
echo "═══════════════════════════════════════════"
echo " source .venv/bin/activate"
echo " python3 train.py"
echo ""
echo " OR without activating (direct uv run):"
echo " uv run python3 train.py"
echo " uv run python3 train.py --resume"
echo " uv run python3 scripts/verify_model.py --model outputs/alpha_functiongemma"
echo "═══════════════════════════════════════════"
