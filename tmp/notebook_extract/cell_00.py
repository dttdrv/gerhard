# =============================================================================
# cell 1: environment setup (v14.1 - Hyperparameter Tuning)
# =============================================================================
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import time
import math
import json
import base64
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# v15 full validation run: disable torch.compile for stability with rich spike instrumentation outputs.
USE_TORCH_COMPILE = False
USE_GRADIENT_CHECKPOINTING = True

# generate timestamp for this run
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M%S')
print(f"run timestamp: {RUN_TIMESTAMP}")

# detect platform
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
IS_COLAB = 'COLAB_GPU' in os.environ or 'google.colab' in sys.modules
PLATFORM = 'kaggle' if IS_KAGGLE else 'colab' if IS_COLAB else 'local'
OUTPUT_DIR = '/kaggle/working/outputs' if IS_KAGGLE else 'outputs'

for subdir in ['figures', 'checkpoints', 'logs', 'results']:
    os.makedirs(f'{OUTPUT_DIR}/{subdir}', exist_ok=True)

print(f"platform: {PLATFORM}")
print(f"output directory: {OUTPUT_DIR}")
print(f"torch.compile: {'enabled' if USE_TORCH_COMPILE else 'disabled'}")
print(f"gradient checkpointing: {'enabled' if USE_GRADIENT_CHECKPOINTING else 'disabled'}")
