# dyntex_python

## 1. Create & Activate a Python Environment
Choose either Conda or the built-in venv:

### 1.1 Using Conda
```bash
conda create -n dyntex-env python=3.10 -y
conda activate dyntex-env
conda install pip -y
```

### 1.2 Using venv (macOS/Linux)
```bash
python3.10 -m venv dyntex-env
source dyntex-env/bin/activate
pip install --upgrade pip
```

### 1.3 Using venv (Windows PowerShell)
```powershell
python -m venv dyntex-env
.\dyntex-env\Scripts\Activate.ps1
pip install --upgrade pip
```

## 2. Install Package Dependencies

**Install PsychoPy**
```bash
pip install psychopy
```

**Install PyTorch**  
Visit the PyTorch install page and choose your CUDA version. Example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

**Install dyntex in editable mode**
```bash
pip install -e .
```

## 3. Run the Demo GUIs
From the project root:

**Basic GUI**
```bash
python demo/simple_gui.py
```

**Composite GUI**
```bash
python demo/compound_gui.py
```
Press `Esc` in the PsychoPy window to exit.

## 4. Troubleshooting
- **Import errors**: ensure you ran `pip install -e .` so that `dyntex` is on your PYTHONPATH.
- **CUDA issues**: verify your GPU drivers and CUDA toolkit match your PyTorch install.
- **No window appears**: check OpenGL support or run under Xvfb on headless servers.
