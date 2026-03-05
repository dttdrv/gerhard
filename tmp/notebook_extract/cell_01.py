# =============================================================================
# cell 2: pytorch and hardware setup (v14.1)
# =============================================================================
# Dependency bootstrap: fail early with clear errors, auto-install when allowed.
import importlib
import subprocess

AUTO_INSTALL_MISSING_DEPS = os.environ.get('GERHARD_AUTO_INSTALL_DEPS', '1') == '1'

def ensure_dependency(import_name: str, pip_name: str = None, required: bool = True) -> bool:
    pip_target = pip_name or import_name
    try:
        importlib.import_module(import_name)
        return True
    except ModuleNotFoundError as exc:
        if AUTO_INSTALL_MISSING_DEPS:
            print(f"missing dependency '{import_name}', attempting pip install: {pip_target}")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', pip_target])
                importlib.import_module(import_name)
                print(f"installed dependency '{import_name}'")
                return True
            except Exception as install_exc:
                message = (
                    f"failed to install dependency '{import_name}' via pip target '{pip_target}'. "
                    f"set GERHARD_AUTO_INSTALL_DEPS=0 to disable auto-install attempts."
                )
                if required:
                    raise ModuleNotFoundError(message) from install_exc
                print(f"warning: {message}")
                return False
        message = (
            f"missing dependency '{import_name}'. "
            f"install '{pip_target}' or set GERHARD_AUTO_INSTALL_DEPS=1 for automatic install."
        )
        if required:
            raise ModuleNotFoundError(message) from exc
        print(f"warning: {message}")
        return False

# Required dependencies used later in the notebook.
ensure_dependency('tqdm', 'tqdm', required=True)
ensure_dependency('transformers', 'transformers', required=True)
ensure_dependency('datasets', 'datasets', required=True)

# Optional plotting dependency. Training/evaluation can proceed without it.
MATPLOTLIB_AVAILABLE = ensure_dependency('matplotlib', 'matplotlib', required=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
import numpy as np

if MATPLOTLIB_AVAILABLE:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    plt = None
    print("warning: matplotlib unavailable; plot generation will be skipped.")

from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"gpu: {gpu_name}")
    print(f"memory: {gpu_memory:.1f} gb")

# v14: set float32 matmul precision for torch.compile
if USE_TORCH_COMPILE and hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
    print("float32 matmul precision: high (for torch.compile)")

print(f"device: {DEVICE}")
print(f"pytorch: {torch.__version__}")

# check torch.compile availability
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile') and torch.__version__ >= '2.0'
print(f"torch.compile available: {TORCH_COMPILE_AVAILABLE}")
