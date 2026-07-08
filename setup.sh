#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "[setup] Creating virtual environment..."
python3.10 -m venv .venv

echo "[setup] Upgrading pip..."
.venv/bin/pip install --upgrade pip

echo "[setup] Installing dependencies..."
.venv/bin/pip install -r requirements.txt

echo ""
echo "[setup] Done. Activate with:  source .venv/bin/activate"
echo "[setup] For GPU support, see the comment in requirements.txt"
