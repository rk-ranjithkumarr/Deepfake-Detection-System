@echo off
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch, torchvision; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo Setup complete. Activate with: venv\Scripts\activate
pause
