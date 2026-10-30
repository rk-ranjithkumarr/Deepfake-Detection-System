#!/usr/bin/env bash
set -e
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch, torchvision; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo "Setup complete. Activate with: source venv/bin/activate"
