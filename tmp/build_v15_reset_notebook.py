import json
from pathlib import Path
from textwrap import dedent

nb_path = Path(r"C:\Users\deyan\Projects\gerhard\notebooks\asnn_goose_v15_reset_master.ipynb")


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }

cells = []

cells.append(md("""
# ASNN-Goose v15 Reset Master

## 1. Purpose / Phase / Expected Outputs
- Purpose: deterministic, notebook-first v15 rerun aligned to the phase lock `v15 -> v16 -> v17 -> v18 -> v19`.
- Phase: `B` (SpikingBrain validation).
- Scope guardrails: no RL, no retrieval-default-path changes, no scaling work, no v16+ execution.
- Expected canonical human-facing output: `outputs/<run_id>/run_dossier_<run_id>.html`.
- Required machine artifacts under `outputs/<run_id>/`:
  - `config.yaml`
  - `seed.txt`
  - `metrics.json`
  - `eval_suite.json`
  - `v15_spikingbrain.json`
"""))

cells.append(code("""
# 2. Repo-state audit summary

INSPECTED_FILES = [
    "knowledge/roadmap.md",
    "changelog.md",
    "docs/ops/STATUS_BOARD.md",
    "docs/ops/GATE_POLICY.md",
    "docs/ops/REPORTING_CONTRACT.md",
    "docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md",
    "state/gate_results.yaml",
    "reports/index.md",
    "notebooks/asnn_goose_colab_v15.ipynb",
    "src/evaluation/spiking_brain.py",
    "src/evaluation/spike_analysis.py",
    "src/models/asnn_goose.py",
    "scripts/register_notebook_run.py",
    "scripts/register_dossier_run.py",
    "scripts/generate_run_dossier.py",
]

repo_audit_summary = {
    "what_already_exists": [
        "Notebook-first ops contract with register_notebook_run/register_dossier_run ingestion.",
        "Phase B gate policy and reporting contract already documented and enforced.",
        "Existing v15 notebook already emits canonical artifact bundle fields and a dossier file.",
    ],
    "what_is_already_patched": [
        "Current v15 notebook includes return_spike_info-safe forward paths and validator guards.",
        "Single-file dossier generation and optional register_run integration are already implemented.",
        "Status board and RunPod handoff docs document current v15 red-gate state and artifact flow.",
    ],
    "what_is_still_red_or_blocked": [
        "Latest run v15_2026-02-23_200258 remains red on phase_b_scientific_thresholds.",
        "Observed blockers: mutual_information=0.0435 (threshold >0.10), cka_mean=0.0196 (threshold >0.30).",
    ],
    "what_this_notebook_changes": [
        "Introduces a clean reset notebook with explicit 14-step execution contract.",
        "Adds mandatory control suite (rate-only, time-shuffle, sign-shuffle, teacher-shuffle) in the same run.",
        "Enforces final canonical dossier filename run_dossier_<run_id>.html and final autopilot decision line.",
    ],
    "observed_local_modifications_before_authoring": [
        "docs/ops/STATUS_BOARD.md",
        "docs/ops/RUNPOD_NOTEBOOK_HANDOFF.md",
        "notebooks/asnn_goose_colab_v15.ipynb",
    ],
}

print("Repo audit complete. Files inspected:")
for p in INSPECTED_FILES:
    print(f"  - {p}")

print("\nAudit summary:")
for section, items in repo_audit_summary.items():
    print(f"\n[{section}]")
    for item in items:
        print(f"  - {item}")
"""))

cells.append(code("""
# 3. Environment bootstrap

import os
import sys
import json
import math
import time
import base64
import random
import importlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from html import escape as html_escape
from io import BytesIO

os.environ["TOKENIZERS_PARALLELISM"] = "false"

AUTO_INSTALL_MISSING_DEPS = os.environ.get("GERHARD_AUTO_INSTALL_DEPS", "1") == "1"


def ensure_dependency(import_name: str, pip_name: Optional[str] = None, required: bool = True) -> bool:
    pip_target = pip_name or import_name
    try:
        importlib.import_module(import_name)
        return True
    except ModuleNotFoundError as exc:
        if AUTO_INSTALL_MISSING_DEPS:
            print(f"Missing dependency '{import_name}', attempting install: {pip_target}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_target])
                importlib.import_module(import_name)
                print(f"Installed dependency '{import_name}'")
                return True
            except Exception as install_exc:
                msg = (
                    f"Failed to install dependency '{import_name}' via '{pip_target}'. "
                    "Set GERHARD_AUTO_INSTALL_DEPS=0 to disable auto-install."
                )
                if required:
                    raise ModuleNotFoundError(msg) from install_exc
                print(f"warning: {msg}")
                return False
        msg = f"Missing dependency '{import_name}'. Install '{pip_target}' first."
        if required:
            raise ModuleNotFoundError(msg) from exc
        print(f"warning: {msg}")
        return False


ensure_dependency("numpy", "numpy", required=True)
ensure_dependency("torch", "torch", required=True)
ensure_dependency("tqdm", "tqdm", required=True)
ensure_dependency("transformers", "transformers", required=True)
ensure_dependency("datasets", "datasets", required=True)
ensure_dependency("yaml", "pyyaml", required=True)
MATPLOTLIB_AVAILABLE = ensure_dependency("matplotlib", "matplotlib", required=False)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import yaml

if MATPLOTLIB_AVAILABLE:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:
    plt = None

NOTEBOOK_NAME = "asnn_goose_v15_reset_master.ipynb"
print("Environment bootstrap complete.")
"""))

cells.append(code("""
# 4. Deterministic config block (required top-level knobs)

RUN_MODE = os.environ.get("GERHARD_RUN_MODE", "SMOKE").strip().upper()
if RUN_MODE not in {"SMOKE", "FULL", "DIAGNOSE"}:
    raise ValueError(f"RUN_MODE must be one of SMOKE/FULL/DIAGNOSE, got: {RUN_MODE}")

RUN_ID = os.environ.get(
    "GERHARD_RUN_ID",
    f"v15_reset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
)
PHASE = "B"
TARGET_GPU_PROFILE = "RTX6000ADA_48GB"
SEED = int(os.environ.get("GERHARD_SEED", "42"))
OUTPUT_ROOT = os.environ.get("GERHARD_OUTPUT_ROOT", "outputs")
ENABLE_REGISTER_RUN = os.environ.get("GERHARD_ENABLE_REGISTER_RUN", "0") == "1"
ENABLE_DOSSIER_EXPORT = os.environ.get("GERHARD_ENABLE_DOSSIER_EXPORT", "1") == "1"
ENABLE_AUTODOWNLOAD_DOSSIER = os.environ.get("GERHARD_ENABLE_AUTODOWNLOAD_DOSSIER", "0") == "1"

CHECKPOINT_PATH = os.environ.get("GERHARD_CHECKPOINT_PATH", "").strip()
DATASET_NAME = os.environ.get("GERHARD_DATASET_NAME", "wikitext")
DATASET_CONFIG = os.environ.get("GERHARD_DATASET_CONFIG", "wikitext-2-raw-v1")
MAX_SEQ_LEN = int(os.environ.get("GERHARD_MAX_SEQ_LEN", "256"))
BATCH_SIZE = int(os.environ.get("GERHARD_BATCH_SIZE", "8"))
SMOKE_BATCHES = int(os.environ.get("GERHARD_SMOKE_BATCHES", "2"))
MODE_TO_VALIDATION_BATCHES = {
    "SMOKE": int(os.environ.get("GERHARD_FULL_BATCHES_SMOKE", "4")),
    "FULL": int(os.environ.get("GERHARD_FULL_BATCHES", "20")),
    "DIAGNOSE": int(os.environ.get("GERHARD_FULL_BATCHES_DIAGNOSE", "40")),
}
MAX_VALIDATION_BATCHES = MODE_TO_VALIDATION_BATCHES[RUN_MODE]

USE_TORCH_COMPILE = False
USE_GRADIENT_CHECKPOINTING = False

# v15 reset policy: deterministic and conservative by default.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_DIR = Path(OUTPUT_ROOT) / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

runtime_knobs = {
    "RUN_MODE": RUN_MODE,
    "RUN_ID": RUN_ID,
    "PHASE": PHASE,
    "TARGET_GPU_PROFILE": TARGET_GPU_PROFILE,
    "SEED": SEED,
    "OUTPUT_ROOT": OUTPUT_ROOT,
    "ENABLE_REGISTER_RUN": ENABLE_REGISTER_RUN,
    "ENABLE_DOSSIER_EXPORT": ENABLE_DOSSIER_EXPORT,
    "ENABLE_AUTODOWNLOAD_DOSSIER": ENABLE_AUTODOWNLOAD_DOSSIER,
    "CHECKPOINT_PATH": CHECKPOINT_PATH,
    "USE_TORCH_COMPILE": USE_TORCH_COMPILE,
    "USE_GRADIENT_CHECKPOINTING": USE_GRADIENT_CHECKPOINTING,
}

print("Runtime knobs:")
for k, v in runtime_knobs.items():
    print(f"  {k}: {v}")

if not ENABLE_DOSSIER_EXPORT:
    raise RuntimeError("ENABLE_DOSSIER_EXPORT must be enabled for v15 reset canonical reporting.")
"""))

cells.append(code("""
# 5. Runtime hardware logging

hardware_info = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "device": str(DEVICE),
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "target_gpu_profile": TARGET_GPU_PROFILE,
}

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = props.total_memory / (1024 ** 3)
    hardware_info.update({
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(float(vram_gb), 2),
        "cuda_version": torch.version.cuda,
        "matches_target_profile": (
            "RTX 6000" in gpu_name.upper() and "ADA" in gpu_name.upper() and vram_gb >= 46.0
        ),
    })
else:
    hardware_info.update({
        "gpu_name": "cpu_only",
        "gpu_vram_gb": 0.0,
        "cuda_version": None,
        "matches_target_profile": False,
    })

print("Hardware info:")
for k, v in hardware_info.items():
    print(f"  {k}: {v}")
"""))

cells.append(code("""
# 6. Data/checkpoint load (fail-fast discipline)

if not CHECKPOINT_PATH:
    raise FileNotFoundError(
        "Missing CHECKPOINT_PATH. Set GERHARD_CHECKPOINT_PATH to a v14.3/v15-compatible student checkpoint."
    )

checkpoint_file = Path(CHECKPOINT_PATH)
if not checkpoint_file.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

print(f"Loading checkpoint: {checkpoint_file}")
checkpoint_obj = torch.load(checkpoint_file, map_location="cpu")


def extract_student_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "student" in obj and isinstance(obj["student"], dict):
            return obj["student"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Unsupported checkpoint format: could not extract student state_dict.")


def infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    embed_key = "embed.weight"
    pos_key = "pos_embed.weight"
    if embed_key not in state_dict or pos_key not in state_dict:
        raise ValueError("State dict missing embed/pos_embed keys required for StudentSpikingGoose.")

    vocab_size, d_model = state_dict[embed_key].shape
    max_seq_len = state_dict[pos_key].shape[0]

    layer_ids = []
    for k in state_dict:
        if k.startswith("layers."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.append(int(parts[1]))
    if not layer_ids:
        raise ValueError("Could not infer n_layers from checkpoint state dict.")

    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "max_seq_len": int(max_seq_len),
        "n_layers": int(max(layer_ids) + 1),
    }


class TrainableTernarySpike(nn.Module):
    def __init__(self, alpha: float = 1.0, threshold_mix: float = 0.35, surrogate_temp: float = 0.10):
        super().__init__()
        self.alpha = alpha
        self.threshold_mix = threshold_mix
        self.surrogate_temp = surrogate_temp
        self.amplitude = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        token_scale = x.abs().mean(dim=-1, keepdim=True)
        channel_scale = x.abs().mean(dim=(0, 1), keepdim=True)
        threshold = self.alpha * (
            (1.0 - self.threshold_mix) * token_scale + self.threshold_mix * channel_scale
        )
        threshold = threshold.clamp(min=0.01, max=10.0)

        with torch.no_grad():
            pos_mask = (x > threshold).float()
            neg_mask = (x < -threshold).float()
            spike_signs = pos_mask - neg_mask

        amplitude = self.amplitude.clamp(min=0.25, max=4.0)
        spikes = amplitude * spike_signs
        spikes = spikes + (x - x.detach())

        if return_aux:
            soft_activity = torch.sigmoid((x.abs() - threshold) / self.surrogate_temp)
            return spikes, {"threshold": threshold.detach(), "soft_activity": soft_activity}
        return spikes

    def get_amplitude(self) -> float:
        return float(self.amplitude.item())


class GooseFFN(nn.Module):
    def __init__(self, d_model: int, expand: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_model * expand, bias=False)
        self.w2 = nn.Linear(d_model * expand, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w2(F.silu(self.w1(self.ln(x))))


class SpikingGooseRecurrentLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        layer_idx: int = 0,
        n_layers: int = 4,
        spike_alpha: float = 1.0,
        threshold_mix: float = 0.35,
        surrogate_temp: float = 0.10,
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.ln = nn.LayerNorm(d_model)

        ratio = layer_idx / max(n_layers - 1, 1)
        self.time_mix_k = nn.Parameter(torch.ones(d_model) * (1 - ratio))
        self.time_mix_v = nn.Parameter(torch.ones(d_model) * (1 - ratio))
        self.time_mix_r = nn.Parameter(torch.ones(d_model) * (1 - ratio))
        self.decay_weight = nn.Parameter(torch.zeros(d_model) - 0.5)

        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.receptance_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

        self.k_spike = TrainableTernarySpike(alpha=spike_alpha, threshold_mix=threshold_mix, surrogate_temp=surrogate_temp)
        self.v_spike = TrainableTernarySpike(alpha=spike_alpha, threshold_mix=threshold_mix, surrogate_temp=surrogate_temp)

        self.register_buffer("running_k_density", torch.tensor(0.0))
        self.register_buffer("running_v_density", torch.tensor(0.0))
        self._init_weights()

    def _init_weights(self):
        std = 0.1 / math.sqrt(self.d_model)
        for proj in [self.key_proj, self.value_proj, self.receptance_proj, self.output_proj]:
            nn.init.normal_(proj.weight, std=std)

    def forward(self, x: torch.Tensor, return_spikes: bool = False, detach_spikes: bool = True):
        _, T, _ = x.shape
        x_norm = self.ln(x)
        prev_x = F.pad(x_norm[:, :-1, :], (0, 0, 1, 0))

        xk = x_norm * self.time_mix_k + prev_x * (1 - self.time_mix_k)
        xv = x_norm * self.time_mix_v + prev_x * (1 - self.time_mix_v)
        xr = x_norm * self.time_mix_r + prev_x * (1 - self.time_mix_r)

        k_pre = self.key_proj(xk)
        v_pre = self.value_proj(xv)

        if return_spikes:
            k, _ = self.k_spike(k_pre, return_aux=True)
            v, _ = self.v_spike(v_pre, return_aux=True)
        else:
            k = self.k_spike(k_pre)
            v = self.v_spike(v_pre)

        r = torch.sigmoid(self.receptance_proj(xr))

        kv = k * v
        decay = torch.sigmoid(self.decay_weight)
        t_idx = torch.arange(T, device=x.device, dtype=x.dtype)
        decay_powers = decay.unsqueeze(0) ** t_idx.unsqueeze(1)

        kv_weighted = kv / (decay_powers.unsqueeze(0) + 1e-8)
        state = torch.cumsum(kv_weighted, dim=1) * decay_powers.unsqueeze(0)

        if self.training:
            with torch.no_grad():
                self.running_k_density = 0.99 * self.running_k_density + 0.01 * (k != 0).float().mean()
                self.running_v_density = 0.99 * self.running_v_density + 0.01 * (v != 0).float().mean()

        out = x + r * self.output_proj(state)

        if return_spikes:
            k_out = k.detach() if detach_spikes else k
            v_out = v.detach() if detach_spikes else v
            return out, {"k_spikes": k_out, "v_spikes": v_out}
        return out

    def get_spike_density(self) -> Dict[str, float]:
        return {
            "k": float(self.running_k_density.item()),
            "v": float(self.running_v_density.item()),
            "k_amp": float(self.k_spike.get_amplitude()),
            "v_amp": float(self.v_spike.get_amplitude()),
        }


class StudentSpikingGoose(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        vocab_size: int,
        max_seq_len: int,
        spike_alpha: float = 1.0,
        spike_threshold_mix: float = 0.35,
        spike_surrogate_temp: float = 0.10,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_checkpointing = use_checkpointing

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "rec": SpikingGooseRecurrentLayer(
                    d_model=d_model,
                    layer_idx=i,
                    n_layers=n_layers,
                    spike_alpha=spike_alpha,
                    threshold_mix=spike_threshold_mix,
                    surrogate_temp=spike_surrogate_temp,
                ),
                "ffn": GooseFFN(d_model=d_model),
            })
            for i in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def _layer_forward(self, layer: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
        x = layer["rec"](x)
        x = layer["ffn"](x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hiddens: bool = False,
        return_spike_info: bool = False,
        detach_spikes: bool = True,
    ):
        _, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)

        hiddens = [x] if return_hiddens else None
        spike_info = {} if return_spike_info else None

        for layer_idx, layer in enumerate(self.layers):
            if self.use_checkpointing and self.training and not return_spike_info:
                x = checkpoint(self._layer_forward, layer, x, use_reentrant=False)
            else:
                if return_spike_info:
                    x, layer_spikes = layer["rec"](x, return_spikes=True, detach_spikes=detach_spikes)
                    x = layer["ffn"](x)
                    spike_info[layer_idx] = layer_spikes
                else:
                    x = self._layer_forward(layer, x)

            if return_hiddens:
                hiddens.append(x)

        logits = self.head(self.ln_out(x))

        if return_hiddens and return_spike_info:
            return logits, hiddens, {"spike_info": spike_info}
        if return_hiddens:
            return logits, hiddens
        if return_spike_info:
            return logits, {"spike_info": spike_info}
        return logits

    def get_avg_spike_density(self) -> float:
        values: List[float] = []
        for layer in self.layers:
            d = layer["rec"].get_spike_density()
            values.extend([d["k"], d["v"]])
        return float(np.mean(values)) if values else 0.0


student_state_dict = extract_student_state_dict(checkpoint_obj)
arch_cfg = infer_arch_from_state_dict(student_state_dict)
print("Inferred architecture from checkpoint:", arch_cfg)

student = StudentSpikingGoose(
    d_model=arch_cfg["d_model"],
    n_layers=arch_cfg["n_layers"],
    vocab_size=arch_cfg["vocab_size"],
    max_seq_len=arch_cfg["max_seq_len"],
    spike_alpha=float(os.environ.get("GERHARD_SPIKE_ALPHA", "1.0")),
    spike_threshold_mix=float(os.environ.get("GERHARD_SPIKE_THRESHOLD_MIX", "0.35")),
    spike_surrogate_temp=float(os.environ.get("GERHARD_SPIKE_SURROGATE_TEMP", "0.10")),
    use_checkpointing=False,
).to(DEVICE)

load_result = student.load_state_dict(student_state_dict, strict=True)
print("Loaded checkpoint into student model.")
print(load_result)
student.eval()
for p in student.parameters():
    p.requires_grad = False

print("Loading GPT-2 teacher...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
teacher.config.use_cache = False
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False


def pre_tokenize(texts: List[str], max_len: int) -> torch.Tensor:
    all_tokens: List[int] = []
    for text in tqdm(texts, desc="tokenizing", leave=False):
        if isinstance(text, str) and text.strip():
            all_tokens.extend(tokenizer.encode(text, max_length=max_len * 2, truncation=True))
    chunks = [
        all_tokens[i:i + max_len]
        for i in range(0, len(all_tokens) - max_len + 1, max(1, max_len // 2))
        if len(all_tokens[i:i + max_len]) == max_len
    ]
    if not chunks:
        raise RuntimeError("Tokenization produced zero chunks. Loader would be invalid.")
    return torch.tensor(chunks, dtype=torch.long)


print(f"Loading dataset: {DATASET_NAME}/{DATASET_CONFIG}")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
val_tokens = pre_tokenize(dataset["validation"]["text"], MAX_SEQ_LEN)

val_loader = DataLoader(
    TensorDataset(val_tokens),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)

if len(val_loader) == 0:
    raise RuntimeError("val_loader is empty. Cannot run v15 validation.")


def extract_input_ids(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        if "input_ids" not in batch:
            raise KeyError("Batch dict missing 'input_ids'.")
        ids = batch["input_ids"]
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Batch tuple/list is empty.")
        ids = batch[0]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if not isinstance(ids, torch.Tensor):
        raise TypeError(f"Extracted input_ids is not a tensor: {type(ids)}")
    if ids.numel() == 0:
        raise ValueError("Extracted input_ids has zero tokens.")
    return ids


print(f"Validation loader ready: {len(val_loader)} batches, batch_size={BATCH_SIZE}")
"""))

cells.append(code("""
# 7. Instrumentation smoke pass

smoke_report: Dict[str, Any] = {
    "checked_batches": 0,
    "spike_layers_seen": set(),
    "non_finite_detected": False,
}

for batch_idx, batch in enumerate(val_loader):
    if batch_idx >= SMOKE_BATCHES:
        break

    input_ids = extract_input_ids(batch).to(DEVICE, non_blocking=True)

    student_out = student(input_ids, return_spike_info=True, detach_spikes=False)
    if not (isinstance(student_out, tuple) and len(student_out) == 2):
        raise RuntimeError("Student smoke forward did not return expected (logits, aux) tuple.")

    student_logits, aux = student_out
    spike_info = aux.get("spike_info", {}) if isinstance(aux, dict) else {}
    if not spike_info:
        raise RuntimeError("Smoke pass collected no spike_info; expected k/v spike traces.")

    for layer_idx, layer_spikes in spike_info.items():
        if not isinstance(layer_spikes, dict):
            raise RuntimeError(f"Spike info for layer {layer_idx} must be dict, got {type(layer_spikes)}")
        if "k_spikes" not in layer_spikes or "v_spikes" not in layer_spikes:
            raise RuntimeError(f"Spike info layer {layer_idx} missing k_spikes/v_spikes.")
        smoke_report["spike_layers_seen"].add(int(layer_idx))

    if not torch.isfinite(student_logits).all():
        smoke_report["non_finite_detected"] = True

    with torch.no_grad():
        teacher_out = teacher(input_ids, output_hidden_states=True)
    if not hasattr(teacher_out, "hidden_states") or teacher_out.hidden_states is None:
        raise RuntimeError("Teacher smoke forward missing hidden_states.")

    smoke_report["checked_batches"] += 1

if smoke_report["checked_batches"] == 0:
    raise RuntimeError("Smoke pass processed zero batches.")
if smoke_report["non_finite_detected"]:
    raise RuntimeError("Smoke pass detected NaN/Inf logits.")

smoke_report["spike_layers_seen"] = sorted(smoke_report["spike_layers_seen"])
print("Smoke pass report:")
print(json.dumps(smoke_report, indent=2))
"""))

cells.append(code("""
# 8. Full v15 validation pass

LAYER_MAP = {0: 2, 2: 7, 4: 11}
THRESHOLDS = {
    "dead_neuron_pct_lt": 0.05,
    "saturated_neuron_pct_lt": 0.10,
    "firing_rate_mean_range": [0.20, 0.60],
    "mutual_information_gt": 0.10,
    "cka_mean_gt": 0.30,
}


def collect_representations(
    student_model: nn.Module,
    teacher_model: nn.Module,
    loader: DataLoader,
    layer_map: Dict[int, int],
    max_batches: int,
) -> Dict[str, Any]:
    student_model.eval()
    teacher_model.eval()

    collected_spikes: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    collected_teacher: Dict[int, List[torch.Tensor]] = {}
    batches_seen = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        ids = extract_input_ids(batch).to(DEVICE, non_blocking=True)
        student_logits, student_aux = student_model(ids, return_spike_info=True, detach_spikes=True)
        if not torch.isfinite(student_logits).all():
            raise RuntimeError(f"Non-finite student logits in validation batch {batch_idx}.")

        spike_info = student_aux.get("spike_info", {}) if isinstance(student_aux, dict) else {}

        for s_layer, spikes in spike_info.items():
            if s_layer not in collected_spikes:
                collected_spikes[s_layer] = {"k": [], "v": []}

            if isinstance(spikes, dict):
                if "k_spikes" in spikes:
                    collected_spikes[s_layer]["k"].append(spikes["k_spikes"].detach().cpu())
                if "v_spikes" in spikes:
                    collected_spikes[s_layer]["v"].append(spikes["v_spikes"].detach().cpu())
            elif isinstance(spikes, list):
                for step_item in spikes:
                    if isinstance(step_item, dict):
                        if "k_spikes" in step_item:
                            collected_spikes[s_layer]["k"].append(step_item["k_spikes"].detach().cpu())
                        if "v_spikes" in step_item:
                            collected_spikes[s_layer]["v"].append(step_item["v_spikes"].detach().cpu())

        with torch.no_grad():
            t_out = teacher_model(ids, output_hidden_states=True)
        if t_out.hidden_states is None:
            raise RuntimeError("Teacher hidden_states missing during validation collection.")

        for t_layer in layer_map.values():
            if t_layer not in collected_teacher:
                collected_teacher[t_layer] = []
            collected_teacher[t_layer].append(t_out.hidden_states[t_layer + 1].detach().cpu())

        batches_seen += 1

    if batches_seen == 0:
        raise RuntimeError("Validation collection saw zero batches.")

    return {
        "spikes": collected_spikes,
        "teacher": collected_teacher,
        "batches_seen": batches_seen,
    }


def compute_health_metrics(collected_spikes: Dict[int, Dict[str, List[torch.Tensor]]]) -> Dict[str, Any]:
    per_channel_rates: Dict[str, np.ndarray] = {}
    all_rates: List[np.ndarray] = []
    all_spike_tensors: List[torch.Tensor] = []

    for layer_idx, data in collected_spikes.items():
        k_list = data.get("k", [])
        v_list = data.get("v", [])
        if not k_list and not v_list:
            continue

        stacked = []
        if k_list:
            k_flat = torch.cat([s.reshape(-1, s.shape[-1]) for s in k_list], dim=0)
            stacked.append(k_flat)
            all_spike_tensors.extend(k_list)
        if v_list:
            v_flat = torch.cat([s.reshape(-1, s.shape[-1]) for s in v_list], dim=0)
            stacked.append(v_flat)
            all_spike_tensors.extend(v_list)

        layer_flat = torch.cat(stacked, dim=0)
        rates = (layer_flat != 0).float().mean(dim=0).numpy()
        per_channel_rates[f"layer_{layer_idx}"] = rates
        all_rates.append(rates)

    if not all_rates:
        return {
            "dead_neuron_pct": 1.0,
            "saturated_neuron_pct": 0.0,
            "firing_rate_mean": 0.0,
            "firing_rate_std": 0.0,
            "per_channel_rates": {},
            "health_pass": False,
            "alerts": ["No spike data collected."],
        }

    all_rates_flat = np.concatenate(all_rates)
    dead_mask = all_rates_flat < 0.001
    dead_pct = float(dead_mask.mean())

    saturated_pct = 0.0
    if all_spike_tensors:
        full_flat = torch.cat([s.reshape(-1, s.shape[-1]) for s in all_spike_tensors], dim=0)
        always_active = (full_flat != 0).all(dim=0).numpy()
        saturated_pct = float(always_active.mean())

    fr_mean = float(all_rates_flat.mean())
    fr_std = float(all_rates_flat.std())

    alerts: List[str] = []
    if dead_pct > THRESHOLDS["dead_neuron_pct_lt"]:
        alerts.append(f"dead_neuron_pct={dead_pct:.4f} > {THRESHOLDS['dead_neuron_pct_lt']:.2f}")
    if saturated_pct >= THRESHOLDS["saturated_neuron_pct_lt"]:
        alerts.append(
            f"saturated_neuron_pct={saturated_pct:.4f} >= {THRESHOLDS['saturated_neuron_pct_lt']:.2f}"
        )
    lo, hi = THRESHOLDS["firing_rate_mean_range"]
    if not (lo <= fr_mean <= hi):
        alerts.append(f"firing_rate_mean={fr_mean:.4f} outside [{lo:.2f}, {hi:.2f}]")

    return {
        "dead_neuron_pct": dead_pct,
        "saturated_neuron_pct": saturated_pct,
        "firing_rate_mean": fr_mean,
        "firing_rate_std": fr_std,
        "per_channel_rates": per_channel_rates,
        "health_pass": len(alerts) == 0,
        "alerts": alerts,
    }


def estimate_mutual_information(spikes: torch.Tensor, teacher_hidden: torch.Tensor, n_bins: int = 32, n_dims: int = 8) -> float:
    s_flat = spikes.reshape(-1, spikes.shape[-1]).float().cpu().numpy()
    t_flat = teacher_hidden.reshape(-1, teacher_hidden.shape[-1]).float().cpu().numpy()

    n = min(s_flat.shape[0], t_flat.shape[0], 10000)
    if n == 0:
        return 0.0

    s_flat = s_flat[:n]
    t_flat = t_flat[:n]
    d = min(n_dims, s_flat.shape[1], t_flat.shape[1])
    if d == 0:
        return 0.0

    mi_values: List[float] = []
    for dim in range(d):
        s_col = s_flat[:, dim]
        t_col = t_flat[:, dim]

        t_min, t_max = float(t_col.min()), float(t_col.max())
        if abs(t_max - t_min) < 1e-12:
            continue

        t_bins = np.digitize(t_col, np.linspace(t_min, t_max, n_bins + 1)[1:-1])

        # Robust ternary discretization by sign only.
        s_disc = np.ones_like(s_col, dtype=np.int32)
        s_disc[s_col > 1e-6] = 2
        s_disc[s_col < -1e-6] = 0

        joint = np.zeros((3, n_bins), dtype=np.float64)
        for i in range(n):
            joint[s_disc[i], max(0, min(int(t_bins[i]), n_bins - 1))] += 1.0

        joint = joint / (joint.sum() + 1e-12)
        p_s = joint.sum(axis=1, keepdims=True) + 1e-12
        p_t = joint.sum(axis=0, keepdims=True) + 1e-12

        mi = float(np.sum(joint * np.log2((joint + 1e-12) / (p_s * p_t))))
        mi_values.append(max(0.0, mi))

    if not mi_values:
        return 0.0
    return float(np.mean(mi_values))


def linear_cka(spikes: torch.Tensor, teacher_hidden: torch.Tensor, max_samples: int = 5000) -> float:
    x = spikes.reshape(-1, spikes.shape[-1]).float().cpu().numpy()
    y = teacher_hidden.reshape(-1, teacher_hidden.shape[-1]).float().cpu().numpy()

    n = min(x.shape[0], y.shape[0], max_samples)
    if n == 0:
        return 0.0

    x = x[:n]
    y = y[:n]

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    hsic_xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(x.T @ x, ord="fro") ** 2
    hsic_yy = np.linalg.norm(y.T @ y, ord="fro") ** 2

    denom = math.sqrt(float(hsic_xx * hsic_yy)) + 1e-12
    return float(hsic_xy / denom)


def flatten_layer_spikes(layer_data: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for t in layer_data.get("k", []):
        chunks.append(t.reshape(-1, t.shape[-1]))
    for t in layer_data.get("v", []):
        chunks.append(t.reshape(-1, t.shape[-1]))
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks, dim=0)


def flatten_teacher_hidden(t_list: List[torch.Tensor], repeat_factor: int = 1) -> torch.Tensor:
    base = torch.cat(t_list, dim=0).reshape(-1, t_list[0].shape[-1])
    if repeat_factor == 1:
        return base
    return torch.cat([base for _ in range(repeat_factor)], dim=0)


def compute_information_and_cka(collected: Dict[str, Any], layer_map: Dict[int, int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mi_per_layer: Dict[str, float] = {}
    cka_per_layer: Dict[str, float] = {}

    for s_layer, t_layer in layer_map.items():
        if s_layer not in collected["spikes"] or t_layer not in collected["teacher"]:
            continue

        layer_spikes = collected["spikes"][s_layer]
        k_list = layer_spikes.get("k", [])
        v_list = layer_spikes.get("v", [])
        t_hidden = torch.cat(collected["teacher"][t_layer], dim=0)

        local_mi: List[float] = []
        local_cka: List[float] = []

        if k_list:
            k_flat = torch.cat(k_list, dim=0)
            m = estimate_mutual_information(k_flat, t_hidden)
            c = linear_cka(k_flat, t_hidden)
            mi_per_layer[f"layer_{s_layer}_to_{t_layer}_k"] = m
            cka_per_layer[f"layer_{s_layer}_to_{t_layer}_k"] = c
            local_mi.append(m)
            local_cka.append(c)

        if v_list:
            v_flat = torch.cat(v_list, dim=0)
            m = estimate_mutual_information(v_flat, t_hidden)
            c = linear_cka(v_flat, t_hidden)
            mi_per_layer[f"layer_{s_layer}_to_{t_layer}_v"] = m
            cka_per_layer[f"layer_{s_layer}_to_{t_layer}_v"] = c
            local_mi.append(m)
            local_cka.append(c)

        if local_mi:
            mi_per_layer[f"layer_{s_layer}_to_{t_layer}"] = float(np.mean(local_mi))
        if local_cka:
            cka_per_layer[f"layer_{s_layer}_to_{t_layer}"] = float(np.mean(local_cka))

    mi_mean = float(np.mean(list(mi_per_layer.values()))) if mi_per_layer else 0.0
    cka_mean = float(np.mean(list(cka_per_layer.values()))) if cka_per_layer else 0.0

    return (
        {**mi_per_layer, "mutual_information": mi_mean, "method": "binning_sign_discretization"},
        {**cka_per_layer, "cka_mean": cka_mean, "method": "linear_cka"},
    )


COLLECTED = collect_representations(
    student_model=student,
    teacher_model=teacher,
    loader=val_loader,
    layer_map=LAYER_MAP,
    max_batches=MAX_VALIDATION_BATCHES,
)

health_metrics = compute_health_metrics(COLLECTED["spikes"])
mutual_information_metrics, cka_metrics = compute_information_and_cka(COLLECTED, LAYER_MAP)

v15_threshold_pass = {
    "dead_neuron_pct": health_metrics["dead_neuron_pct"] < THRESHOLDS["dead_neuron_pct_lt"],
    "saturated_neuron_pct": health_metrics["saturated_neuron_pct"] < THRESHOLDS["saturated_neuron_pct_lt"],
    "firing_rate_mean": (
        THRESHOLDS["firing_rate_mean_range"][0]
        <= health_metrics["firing_rate_mean"]
        <= THRESHOLDS["firing_rate_mean_range"][1]
    ),
    "mutual_information": mutual_information_metrics.get("mutual_information", 0.0) > THRESHOLDS["mutual_information_gt"],
    "cka_mean": cka_metrics.get("cka_mean", 0.0) > THRESHOLDS["cka_mean_gt"],
}

v15_results = {
    "run_id": RUN_ID,
    "phase": PHASE,
    "validation": {
        "health": health_metrics,
        "mutual_information": mutual_information_metrics,
        "cka": cka_metrics,
    },
    "threshold_pass": v15_threshold_pass,
    "overall_pass_thresholds": bool(all(v15_threshold_pass.values())),
    "max_batches_used": MAX_VALIDATION_BATCHES,
}

print("V15 core validation results:")
print(json.dumps({
    "health": {
        "dead_neuron_pct": health_metrics["dead_neuron_pct"],
        "saturated_neuron_pct": health_metrics["saturated_neuron_pct"],
        "firing_rate_mean": health_metrics["firing_rate_mean"],
    },
    "mutual_information": mutual_information_metrics.get("mutual_information", 0.0),
    "cka_mean": cka_metrics.get("cka_mean", 0.0),
    "overall_pass_thresholds": v15_results["overall_pass_thresholds"],
}, indent=2))
"""))

cells.append(code("""
# 9. Control suite

CONTROL_MI_MARGIN = float(os.environ.get("GERHARD_CONTROL_MI_MARGIN", "0.005"))
CONTROL_CKA_MARGIN = float(os.environ.get("GERHARD_CONTROL_CKA_MARGIN", "0.010"))


def align_lengths(spikes_flat: torch.Tensor, teacher_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = min(spikes_flat.shape[0], teacher_flat.shape[0])
    if n <= 0:
        return spikes_flat[:0], teacher_flat[:0]
    return spikes_flat[:n], teacher_flat[:n]


def make_rate_only_baseline(spikes_flat: torch.Tensor) -> torch.Tensor:
    if spikes_flat.numel() == 0:
        return spikes_flat

    active = (spikes_flat != 0)
    rates = active.float().mean(dim=0, keepdim=True)
    sampled_active = torch.rand_like(spikes_flat.float()) < rates

    nz = spikes_flat[active]
    if nz.numel() == 0:
        return torch.zeros_like(spikes_flat)

    pos_prob = float((nz > 0).float().mean().item())
    sampled_sign = torch.where(
        torch.rand_like(spikes_flat.float()) < pos_prob,
        torch.ones_like(spikes_flat),
        -torch.ones_like(spikes_flat),
    )

    amplitude = float(nz.abs().mean().item())
    out = torch.zeros_like(spikes_flat)
    out[sampled_active] = sampled_sign[sampled_active] * amplitude
    return out


def make_time_shuffle(layer_data: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
    shuffled_chunks: List[torch.Tensor] = []
    for key in ("k", "v"):
        for tensor in layer_data.get(key, []):
            if tensor.ndim != 3:
                shuffled_chunks.append(tensor.reshape(-1, tensor.shape[-1]))
                continue
            B, T, D = tensor.shape
            perm = torch.randperm(T)
            shuffled = tensor[:, perm, :]
            shuffled_chunks.append(shuffled.reshape(-1, D))
    if not shuffled_chunks:
        return torch.empty(0)
    return torch.cat(shuffled_chunks, dim=0)


def make_sign_shuffle(spikes_flat: torch.Tensor) -> torch.Tensor:
    if spikes_flat.numel() == 0:
        return spikes_flat

    out = torch.zeros_like(spikes_flat)
    mask = spikes_flat != 0
    values = spikes_flat[mask]
    if values.numel() == 0:
        return out

    abs_vals = values.abs()
    signs = values.sign()
    perm = torch.randperm(signs.numel())
    shuffled_signs = signs[perm]
    out[mask] = abs_vals * shuffled_signs
    return out


def make_teacher_shuffle(teacher_flat: torch.Tensor) -> torch.Tensor:
    if teacher_flat.numel() == 0:
        return teacher_flat
    perm = torch.randperm(teacher_flat.shape[0])
    return teacher_flat[perm]


def score_control_pair(spikes_flat: torch.Tensor, teacher_flat: torch.Tensor) -> Dict[str, float]:
    spikes_flat, teacher_flat = align_lengths(spikes_flat, teacher_flat)
    if spikes_flat.numel() == 0 or teacher_flat.numel() == 0:
        return {"mutual_information": 0.0, "cka_mean": 0.0}
    return {
        "mutual_information": estimate_mutual_information(spikes_flat, teacher_flat),
        "cka_mean": linear_cka(spikes_flat, teacher_flat),
    }


control_layer_rows: Dict[str, List[Dict[str, Any]]] = {
    "real_spikes_vs_teacher": [],
    "rate_only_baseline": [],
    "time_shuffle_preserving_counts": [],
    "sign_shuffle": [],
    "teacher_shuffle": [],
}

for s_layer, t_layer in LAYER_MAP.items():
    if s_layer not in COLLECTED["spikes"] or t_layer not in COLLECTED["teacher"]:
        continue

    layer_data = COLLECTED["spikes"][s_layer]
    real_spikes = flatten_layer_spikes(layer_data)
    teacher_flat = flatten_teacher_hidden(COLLECTED["teacher"][t_layer], repeat_factor=2)

    controls = {
        "real_spikes_vs_teacher": (real_spikes, teacher_flat),
        "rate_only_baseline": (make_rate_only_baseline(real_spikes), teacher_flat),
        "time_shuffle_preserving_counts": (make_time_shuffle(layer_data), teacher_flat),
        "sign_shuffle": (make_sign_shuffle(real_spikes), teacher_flat),
        "teacher_shuffle": (real_spikes, make_teacher_shuffle(teacher_flat)),
    }

    for control_name, (s_flat, t_flat) in controls.items():
        scores = score_control_pair(s_flat, t_flat)
        control_layer_rows[control_name].append({
            "student_layer": s_layer,
            "teacher_layer": t_layer,
            **scores,
        })


CONTROL_RESULTS: Dict[str, Dict[str, Any]] = {}
for control_name, rows in control_layer_rows.items():
    if rows:
        mi_mean = float(np.mean([r["mutual_information"] for r in rows]))
        cka_mean = float(np.mean([r["cka_mean"] for r in rows]))
    else:
        mi_mean = 0.0
        cka_mean = 0.0

    CONTROL_RESULTS[control_name] = {
        "mutual_information": mi_mean,
        "cka_mean": cka_mean,
        "per_layer": rows,
    }

real_control = CONTROL_RESULTS["real_spikes_vs_teacher"]
CONTROL_SEPARATION = {}
for control_name, payload in CONTROL_RESULTS.items():
    if control_name == "real_spikes_vs_teacher":
        continue
    CONTROL_SEPARATION[control_name] = {
        "mi_delta_vs_real": real_control["mutual_information"] - payload["mutual_information"],
        "cka_delta_vs_real": real_control["cka_mean"] - payload["cka_mean"],
    }

CONTROL_SEPARATION_PASS = all(
    (d["mi_delta_vs_real"] > CONTROL_MI_MARGIN) and (d["cka_delta_vs_real"] > CONTROL_CKA_MARGIN)
    for d in CONTROL_SEPARATION.values()
)

print("Control suite summary:")
print(json.dumps({
    "controls": {k: {"mutual_information": v["mutual_information"], "cka_mean": v["cka_mean"]} for k, v in CONTROL_RESULTS.items()},
    "separation": CONTROL_SEPARATION,
    "separation_pass": CONTROL_SEPARATION_PASS,
    "margins": {"mi_margin": CONTROL_MI_MARGIN, "cka_margin": CONTROL_CKA_MARGIN},
}, indent=2))
"""))

cells.append(code("""
# 10. Gate evaluation

GATE_SCORECARD: List[Dict[str, Any]] = []

health = v15_results["validation"]["health"]
mi = v15_results["validation"]["mutual_information"].get("mutual_information", 0.0)
cka = v15_results["validation"]["cka"].get("cka_mean", 0.0)

checks = [
    (
        "dead_neuron_pct",
        health["dead_neuron_pct"] < THRESHOLDS["dead_neuron_pct_lt"],
        health["dead_neuron_pct"],
        f"< {THRESHOLDS['dead_neuron_pct_lt']}",
    ),
    (
        "saturated_neuron_pct",
        health["saturated_neuron_pct"] < THRESHOLDS["saturated_neuron_pct_lt"],
        health["saturated_neuron_pct"],
        f"< {THRESHOLDS['saturated_neuron_pct_lt']}",
    ),
    (
        "firing_rate_mean",
        THRESHOLDS["firing_rate_mean_range"][0] <= health["firing_rate_mean"] <= THRESHOLDS["firing_rate_mean_range"][1],
        health["firing_rate_mean"],
        f"in [{THRESHOLDS['firing_rate_mean_range'][0]}, {THRESHOLDS['firing_rate_mean_range'][1]}]",
    ),
    (
        "mutual_information",
        mi > THRESHOLDS["mutual_information_gt"],
        mi,
        f"> {THRESHOLDS['mutual_information_gt']}",
    ),
    (
        "cka_mean",
        cka > THRESHOLDS["cka_mean_gt"],
        cka,
        f"> {THRESHOLDS['cka_mean_gt']}",
    ),
    (
        "control_separation",
        CONTROL_SEPARATION_PASS,
        CONTROL_SEPARATION,
        f"mi_delta>{CONTROL_MI_MARGIN}, cka_delta>{CONTROL_CKA_MARGIN} against all controls",
    ),
]

for gate_name, passed, observed, threshold in checks:
    GATE_SCORECARD.append({
        "gate_name": gate_name,
        "status": "green" if passed else "red",
        "observed": observed,
        "threshold": threshold,
    })

AUTOPILOT_DECISION = "CONTINUE" if all(g["status"] == "green" for g in GATE_SCORECARD) else "PAUSE_NEEDS_INPUT"
NEXT_ACTION = (
    "Phase B green. v16 remains next in chain lock."
    if AUTOPILOT_DECISION == "CONTINUE"
    else "Phase B remains blocked. Use red-gate diagnosis notebook only if tighter diagnosis is required."
)

print("Gate scorecard:")
for g in GATE_SCORECARD:
    print(f"  [{g['status'].upper()}] {g['gate_name']}: observed={g['observed']} threshold={g['threshold']}")

print(f"\nProvisional decision: AUTOPILOT_DECISION: {AUTOPILOT_DECISION}")
print(f"Next action: {NEXT_ACTION}")
"""))

cells.append(code("""
# 11. Artifact save

import hashlib


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=_json_default)


config_payload = {
    "run_id": RUN_ID,
    "phase": PHASE,
    "seed": SEED,
    "target_gpu_profile": TARGET_GPU_PROFILE,
    "runtime_knobs": runtime_knobs,
    "hardware_info": hardware_info,
    "dataset": {
        "name": DATASET_NAME,
        "config": DATASET_CONFIG,
        "max_seq_len": MAX_SEQ_LEN,
        "batch_size": BATCH_SIZE,
    },
    "architecture": arch_cfg,
    "checkpoint_path": str(checkpoint_file),
    "repo_audit_summary": repo_audit_summary,
}

config_sha256 = hashlib.sha256(_safe_json_dumps(config_payload).encode("utf-8")).hexdigest()
recipe_payload = {
    "run_mode": RUN_MODE,
    "max_validation_batches": MAX_VALIDATION_BATCHES,
    "smoke_batches": SMOKE_BATCHES,
    "layer_map": LAYER_MAP,
    "thresholds": THRESHOLDS,
    "control_margins": {"mi": CONTROL_MI_MARGIN, "cka": CONTROL_CKA_MARGIN},
}
recipe_sha256 = hashlib.sha256(_safe_json_dumps(recipe_payload).encode("utf-8")).hexdigest()

notebook_candidates = [
    os.environ.get("GERHARD_NOTEBOOK_PATH", ""),
    str(Path.cwd() / "notebooks" / NOTEBOOK_NAME),
    str(Path.cwd() / NOTEBOOK_NAME),
]
notebook_sha256 = None
notebook_path = None
for candidate in notebook_candidates:
    if not candidate:
        continue
    p = Path(candidate)
    if p.exists() and p.is_file():
        notebook_path = str(p.resolve())
        notebook_sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
        break

run_fingerprint = {
    "config_sha256": config_sha256,
    "recipe_sha256": recipe_sha256,
    "notebook_path": notebook_path,
    "notebook_sha256": notebook_sha256,
}

metrics_payload = {
    "run_id": RUN_ID,
    "phase": PHASE,
    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "seed": SEED,
    "fingerprint": run_fingerprint,
    "v15_dead_neuron_pct": float(health["dead_neuron_pct"]),
    "v15_saturated_neuron_pct": float(health["saturated_neuron_pct"]),
    "v15_firing_rate_mean": float(health["firing_rate_mean"]),
    "v15_mutual_information": float(mi),
    "v15_cka_mean": float(cka),
    "v15_thresholds_pass": bool(v15_results["overall_pass_thresholds"]),
    "v15_control_separation_pass": bool(CONTROL_SEPARATION_PASS),
    "autopilot_decision": AUTOPILOT_DECISION,
}

v15_spikingbrain_payload = {
    "run_id": RUN_ID,
    "phase": PHASE,
    "validation": v15_results["validation"],
    "threshold_pass": v15_results["threshold_pass"],
    "overall_pass_thresholds": v15_results["overall_pass_thresholds"],
    "controls": CONTROL_RESULTS,
    "control_separation": CONTROL_SEPARATION,
    "control_separation_pass": CONTROL_SEPARATION_PASS,
    "gate_scorecard": GATE_SCORECARD,
}

eval_suite_payload = {
    "schema_version": "1.0",
    "run_id": RUN_ID,
    "phase": PHASE,
    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "fingerprint": run_fingerprint,
    "tasks": {
        "v15_spike_health": {
            "dead_neuron_pct": float(health["dead_neuron_pct"]),
            "saturated_neuron_pct": float(health["saturated_neuron_pct"]),
            "firing_rate_mean": float(health["firing_rate_mean"]),
        },
        "v15_information": {
            "mutual_information": float(mi),
            "cka_mean": float(cka),
        },
        "v15_controls": {
            k: {
                "mutual_information": float(v["mutual_information"]),
                "cka_mean": float(v["cka_mean"]),
            }
            for k, v in CONTROL_RESULTS.items()
        },
    },
    "gate_scorecard": GATE_SCORECARD,
    "gate_recommendation": "green" if AUTOPILOT_DECISION == "CONTINUE" else "red",
}

(RUN_DIR / "metrics.json").write_text(_safe_json_dumps(metrics_payload), encoding="utf-8")
(RUN_DIR / "eval_suite.json").write_text(_safe_json_dumps(eval_suite_payload), encoding="utf-8")
(RUN_DIR / "v15_spikingbrain.json").write_text(_safe_json_dumps(v15_spikingbrain_payload), encoding="utf-8")
(RUN_DIR / "seed.txt").write_text(str(SEED) + "\n", encoding="utf-8")

config_with_fingerprint = dict(config_payload)
config_with_fingerprint["fingerprint"] = run_fingerprint
try:
    (RUN_DIR / "config.yaml").write_text(
        yaml.safe_dump(config_with_fingerprint, sort_keys=False),
        encoding="utf-8",
    )
except Exception as exc:
    (RUN_DIR / "config.yaml").write_text(_safe_json_dumps(config_with_fingerprint), encoding="utf-8")
    print(f"warning: YAML export failed, JSON fallback written to config.yaml ({exc})")

artifact_manifest = [
    str((RUN_DIR / "config.yaml").as_posix()),
    str((RUN_DIR / "seed.txt").as_posix()),
    str((RUN_DIR / "metrics.json").as_posix()),
    str((RUN_DIR / "eval_suite.json").as_posix()),
    str((RUN_DIR / "v15_spikingbrain.json").as_posix()),
]

print("Artifacts written:")
for p in artifact_manifest:
    print(f"  - {p}")
"""))

cells.append(code("""
# 12. Single-file dossier generation


def render_controls_plot_base64(controls: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not MATPLOTLIB_AVAILABLE:
        return None

    names = list(controls.keys())
    mi_vals = [float(controls[n]["mutual_information"]) for n in names]
    cka_vals = [float(controls[n]["cka_mean"]) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(len(names)), mi_vals, color="tab:blue")
    axes[0].set_title("Mutual Information by Control")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right")

    axes[1].bar(range(len(names)), cka_vals, color="tab:orange")
    axes[1].set_title("CKA by Control")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha="right")

    for ax in axes:
        ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_firing_plot_base64(per_channel_rates: Dict[str, Any]) -> Optional[str]:
    if not MATPLOTLIB_AVAILABLE:
        return None

    values = []
    for arr in per_channel_rates.values():
        if isinstance(arr, np.ndarray):
            values.extend(arr.tolist())
        elif isinstance(arr, list):
            values.extend(arr)

    if not values:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=40, color="tab:green", alpha=0.8)
    ax.axvspan(THRESHOLDS["firing_rate_mean_range"][0], THRESHOLDS["firing_rate_mean_range"][1], alpha=0.15)
    ax.set_title("Per-channel firing-rate distribution")
    ax.set_xlabel("firing_rate")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


controls_plot_b64 = render_controls_plot_base64(CONTROL_RESULTS)
firing_plot_b64 = render_firing_plot_base64(health.get("per_channel_rates", {}))

def _json_block(payload: Any) -> str:
    return html_escape(_safe_json_dumps(payload))

control_rows = "".join(
    f"<tr><td>{html_escape(name)}</td><td>{payload['mutual_information']:.6f}</td><td>{payload['cka_mean']:.6f}</td></tr>"
    for name, payload in CONTROL_RESULTS.items()
)

gate_rows = "".join(
    f"<tr><td>{html_escape(g['gate_name'])}</td><td>{html_escape(str(g['observed']))}</td><td>{html_escape(str(g['threshold']))}</td><td>{html_escape(g['status'])}</td></tr>"
    for g in GATE_SCORECARD
)

manifest_rows = "".join(f"<li><code>{html_escape(path)}</code></li>" for path in artifact_manifest)

plots_html = ""
if controls_plot_b64:
    plots_html += (
        "<h3>Control Comparison Plot</h3>"
        f"<img src='data:image/png;base64,{controls_plot_b64}' style='max-width:100%;border:1px solid #ccc;padding:4px;'/>"
    )
if firing_plot_b64:
    plots_html += (
        "<h3>Firing Rate Plot</h3>"
        f"<img src='data:image/png;base64,{firing_plot_b64}' style='max-width:100%;border:1px solid #ccc;padding:4px;'/>"
    )
if not plots_html:
    plots_html = "<p>Plot rendering unavailable in this runtime (matplotlib missing).</p>"

html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Gerhard v15 run dossier - {html_escape(RUN_ID)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 22px; color: #111; line-height: 1.4; }}
    h1,h2,h3 {{ margin-top: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #cfcfcf; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f2f4f6; }}
    pre {{ background: #f5f7f9; border: 1px solid #ddd; padding: 10px; overflow-x: auto; }}
    code {{ background: #f5f7f9; padding: 1px 4px; }}
    .pill-green {{ color: #0a6a2a; font-weight: bold; }}
    .pill-red {{ color: #a51d2d; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Gerhard v15 Reset Dossier</h1>
  <p>
    <b>Run ID:</b> {html_escape(RUN_ID)}<br/>
    <b>Phase:</b> {html_escape(PHASE)}<br/>
    <b>Decision:</b> <span class=\"{'pill-green' if AUTOPILOT_DECISION == 'CONTINUE' else 'pill-red'}\">AUTOPILOT_DECISION: {html_escape(AUTOPILOT_DECISION)}</span>
  </p>

  <h2>Executive Summary</h2>
  <ul>
    <li>Latest accepted best model baseline: v14.3 (PPL 306.89).</li>
    <li>Current gate context at reset start: Phase B red on MI/CKA.</li>
    <li>This notebook enforces deterministic v15 rerun + control-separation checks.</li>
  </ul>

  <h2>Repo Baseline</h2>
  <pre>{_json_block(repo_audit_summary)}</pre>

  <h2>Hardware</h2>
  <pre>{_json_block(hardware_info)}</pre>

  <h2>Config</h2>
  <pre>{_json_block(config_with_fingerprint)}</pre>

  <h2>Metric Tables</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Threshold</th></tr>
    <tr><td>dead_neuron_pct</td><td>{health['dead_neuron_pct']:.6f}</td><td>&lt; {THRESHOLDS['dead_neuron_pct_lt']}</td></tr>
    <tr><td>saturated_neuron_pct</td><td>{health['saturated_neuron_pct']:.6f}</td><td>&lt; {THRESHOLDS['saturated_neuron_pct_lt']}</td></tr>
    <tr><td>firing_rate_mean</td><td>{health['firing_rate_mean']:.6f}</td><td>[{THRESHOLDS['firing_rate_mean_range'][0]}, {THRESHOLDS['firing_rate_mean_range'][1]}]</td></tr>
    <tr><td>mutual_information</td><td>{mi:.6f}</td><td>&gt; {THRESHOLDS['mutual_information_gt']}</td></tr>
    <tr><td>cka_mean</td><td>{cka:.6f}</td><td>&gt; {THRESHOLDS['cka_mean_gt']}</td></tr>
  </table>

  <h2>Control Comparison Table</h2>
  <table>
    <tr><th>Control</th><th>Mutual Information</th><th>CKA Mean</th></tr>
    {control_rows}
  </table>

  <h2>Plots</h2>
  {plots_html}

  <h2>Gate Verdict</h2>
  <table>
    <tr><th>Gate</th><th>Observed</th><th>Threshold</th><th>Status</th></tr>
    {gate_rows}
  </table>
  <p><b>Next action:</b> {html_escape(NEXT_ACTION)}</p>

  <h2>Artifact Manifest</h2>
  <ul>
    {manifest_rows}
    <li><code>{html_escape(str((RUN_DIR / f'run_dossier_{RUN_ID}.html').as_posix()))}</code></li>
  </ul>

  <h2>Embedded Raw Payloads</h2>
  <details><summary>metrics.json</summary><pre>{_json_block(metrics_payload)}</pre></details>
  <details><summary>eval_suite.json</summary><pre>{_json_block(eval_suite_payload)}</pre></details>
  <details><summary>v15_spikingbrain.json</summary><pre>{_json_block(v15_spikingbrain_payload)}</pre></details>
</body>
</html>
"""

dossier_path = RUN_DIR / f"run_dossier_{RUN_ID}.html"
dossier_path.write_text(html_doc, encoding="utf-8")
print(f"Dossier written: {dossier_path}")

if ENABLE_AUTODOWNLOAD_DOSSIER:
    try:
        if "google.colab" in sys.modules:
            from google.colab import files
            files.download(str(dossier_path))
            print("Colab auto-download attempted.")
        else:
            from IPython.display import Javascript, display
            abs_path = str(dossier_path.resolve()).replace("\\", "/")
            display(Javascript(f"window.open('/files/{abs_path}', '_blank');"))
            print("Notebook auto-open attempted.")
    except Exception as exc:
        print(f"Auto-download/open skipped: {exc}")
"""))

cells.append(code("""
# 13. Optional ingestion call

registration_result = None
registration_error = None

if ENABLE_REGISTER_RUN:
    candidate_roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path("/workspace/gerhard"),
        Path("/kaggle/working/gerhard"),
    ]
    repo_root = None
    for candidate in candidate_roots:
        if (candidate / "scripts" / "register_notebook_run.py").exists():
            repo_root = candidate
            break

    if repo_root is None:
        registration_error = "register_notebook_run.py not found under candidate repo roots."
    else:
        try:
            if str(repo_root) not in sys.path:
                sys.path.append(str(repo_root))
            from scripts.register_notebook_run import register_run

            registration_result = register_run(
                run_id=RUN_ID,
                phase=PHASE,
                source_dir=RUN_DIR,
                repo_root=repo_root,
                summary="v15 reset master notebook run with control suite and canonical dossier",
                next_action="Proceed according to gate decision (continue on green, pause on red).",
            )
        except Exception as exc:
            registration_error = str(exc)

print("registration_result:", registration_result)
print("registration_error:", registration_error)
"""))

cells.append(code("""
# 14. Final line

if AUTOPILOT_DECISION not in {"CONTINUE", "PAUSE_NEEDS_INPUT"}:
    raise RuntimeError(f"Invalid AUTOPILOT_DECISION value: {AUTOPILOT_DECISION}")

print(f"AUTOPILOT_DECISION: {AUTOPILOT_DECISION}")
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

nb_path.parent.mkdir(parents=True, exist_ok=True)
nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote notebook: {nb_path}")
