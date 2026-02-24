# Gerhard — Master Plan (Post‑v15) + Add‑On Integration (2026‑02‑23)

This is the merged plan: the core Gerhard dependency chain (v15→v19) plus the add‑on tracks (generalist scorecard, retrieval/world‑fit, post‑training RL), integrated without breaking sequencing.

Audience: your coding agent (implements), and you (CEO) for governance and “what counts as proof”.

---

## 0) Ground rules (non‑negotiable)

0.1 Core dependency chain is fixed  
v15 (SpikingBrain) → v16 (sparse ops) → v17 (efficiency benchmarks) → v18 (ablations) → v19 (publication).  
Everything else is *scaffolding* until v15 passes.

0.2 “Proof before spending money”  
Every claim must be backed by an artifact:
- a deterministic test, or
- a benchmark JSON, or
- a reproducible report with shuffled controls.

0.3 Every new capability needs 4 things  
Baseline + Ablation + Evaluation gate + Rollback flag.

0.4 Cost hygiene (RunPod reality)  
No GPU debugging. CPU-first tests. GPU only for short smoke runs unless explicitly authorized.

---

## 1) Snapshot of what must not be lost (baseline invariants)

- Student best: v14.3 (74M params, d_model=768, 5 layers), PPL 306.89.
- Teacher GPT‑2 baseline: PPL ~44.6.
- Spike density: ~38% non‑zero (~62% zeros).
- Thesis: ternary spikes (−1,0,+1) enable add/sub/skip; speedup depends on sparse/structured execution, not just values.

These invariants must remain true after refactors:
- output logits identical (or within tight tolerance) when optional instrumentation is disabled,
- no changes to training/eval protocols unless versioned and recorded.

---

## 2) Review verdict on the add‑on (what we accept / modify / defer)

### 2.1 Accept (high alignment with your goals)
A) Unified evaluation harness + dataset registry + verifier interface stubs  
These are low-cost, de-risk later work, and prevent “we improved one thing, broke three others”.

B) Retrieval (“world-fit”) as external memory  
This matches your “brain-like memory” desire, but must be measured end-to-end for latency and must have citation/grounding checks.

C) Post-training hierarchy: Distillation first, RL second  
Strong-to-weak distillation is usually cheaper and more stable than RL; RL should be applied only where reward is deterministic.

D) Scaling ladder concept (74M→300M→1B), gated  
Scaling is allowed only after the evaluation gates exist and the smaller model is stable.

### 2.2 Modify (to prevent budget burn and instability)
A) RL scope: RLVR only (verifiable rewards) for now  
No broad RLHF. Use deterministic verifiers (code execution, math equivalence). Use strict KL constraints and regression gates.

B) Retrieval: offline pack + caching first  
Online retrieval is deferred until you have a stable product story and cost model.

C) Generalist scorecard: keep it small and repeatable  
Avoid bloated benchmark lists. A compact scorecard that’s stable is more valuable than 50 noisy metrics.

### 2.3 Defer (not rejected, but not in the critical path)
- Large-scale preference tuning for style/helpfulness (DPO) until the model can reliably follow instructions.
- 1B scaling until the 74M track has stable v15–v17 artifacts and a credible compute budget.

---

## 3) Repo structure changes (merge immediately; scaffolding only)

These are safe to merge during Phase A without breaking the chain:

- `eval/`  
  - `eval/run_suite.py` (single entrypoint)  
  - `eval/suite.yaml` (tasks + metrics + gates)  
  - outputs: `outputs/eval/eval_suite.json`

- `data/`  
  - `data/mixture.yaml` (datasets + weights + licenses + max context)

- `verifiers/`  
  - `verifiers/base.py` interface: `verify(prompt, completion) -> {reward, meta}`  
  - stubs: `python_math_verifier`, `code_sandbox_verifier`, `retrieval_citation_verifier`

- `retrieval/`  
  - interface stub `retrieve(query) -> passages` (returns empty initially)

All above must include tests and “disabled by default” behavior.

---

## 4) Core phases (v15→v19) with add‑on hooks

## Phase A — Engineering guardrails + scaffolding skeletons (cheap, mandatory)

Deliverables:
- deterministic run harness + smoke tests
- eval/data/verifier/retrieval skeletons (stubs)

A1. Reproducibility discipline  
Every run writes:
- config YAML
- git commit
- package versions
- RNG seed  
and all artifacts are versioned by run_id.

A2. CPU-first test suite  
Minimum tests:
- model forward/backward smoke
- ternary activation value-range checks
- finite loss + non-NaN gradients
- “tracing off = identical logits” tests (for later membrane distillation hooks)

A3. Hard stop safety flags  
All scripts must accept:
- `--max_steps`, `--max_batches`, `--max_seq_len`, `--dry_run`, `--cpu_only`.

A4–A6 (from add‑on) implemented as stubs  
- unified eval harness output JSON
- dataset mixture YAML (even placeholders)
- verifier interface + stubs

---

## Phase B — v15: SpikingBrain validation (gate to v16)

Deliverables:
- `outputs/v15/v15_spikingbrain.json`
- `outputs/v15/report.md` + plots
- pass/fail summary vs thresholds, plus shuffled controls

B1. Fix CKA implementation (memory-safe)  
Implement feature-space linear CKA via XᵀY statistics; stream over tokens to avoid N×N matrices.

B2. Add “no-cheating” controls (required)  
- rate-only baseline
- time-shuffle (preserve per-neuron spike counts, scramble time)
- sign-shuffle
- teacher shuffle  
The report must show: “real” MI/CKA is meaningfully above the controls.

B3. Persist v15 outputs for drift tracking  
This enables later scaling (K phase) to detect “spike health drift” when you increase model size.

Stop condition:
- If v15 fails, you do not proceed to sparse ops. You repair spike semantics first.

Implementation update (2026-02-23):
- `notebooks/asnn_goose_colab_v15.ipynb` was hardened for live runtime stability before gate evaluation.
- Fixes include dependency bootstrap, validator batch compatibility, spike-info forward path, and empty-data guards.
- Phase B gate decision remains pending until the active RunPod run produces canonical artifacts and report ingestion completes.

---

## Phase C — Temporal coding proof suite (parallel validation track)

Purpose:
Prove (in controlled SNN tasks) that your training approach can learn timing-based codes beyond firing rate.

Deliverables:
- `experiments/temporal_coding/` (snnTorch baseline)
- `outputs/temporal_coding/report.md` (accuracy + perturbation curves)
- unit tests that enforce spike-count equality

Tasks:
- order task, interval task, synchrony task  
Controls:
- rate-only baseline near chance
- timing destruction (shuffle times, preserve counts) drops to chance

This is a scientific “proof artifact” that strengthens the project narrative and guides future “timing-sensitive” mechanisms.

---

## Phase D — v16: Sparse operations (correctness first)

Deliverables:
- sparse state update implementation
- correctness tests vs dense baseline (tight tolerance)
- feature flags to enable/disable sparse path

D1. Implement sparse state-update path  
Start with sparse update over dense state: compute non-zero indices for k⊙v and scatter-add into state.

D2. Optional structured sparsity prototype (separate branch)  
Prototype 2:4 semi-structured weight sparsity for the largest Linear layers ONLY.
- correctness tests
- microbench  
Merge only if speedup is real on target GPUs.

---

## Phase E — v17: Efficiency metrics (after sparse ops exist)

Deliverables:
- `outputs/v17/benchmarks.json` + plots
- includes model-only and end-to-end (with retrieval enabled) benchmarks

E1. Model-only benchmarks  
- tokens/sec, ms/token
- peak VRAM
- p50/p95 latency

E2. End-to-end benchmarks (new add-on requirement)  
When retrieval is enabled:
- include retrieval latency
- cache hit rates
- index size  
This is mandatory because retrieval can destroy UX if not measured.

---

## Phase F — v18: Ablations (expanded)

Deliverables:
- ablation matrix + results table + plots
- run scripts that reproduce each ablation with max-step guards

Required ablations:
- KD: CE only vs logit KD vs CTKD vs +FDD vs full
- spike: dense vs binary vs ternary
- retrieval: on vs off (identical prompts)
- post-training: off-policy distill vs on-policy distill; RLVR vs DPO (if run)

New ablation (from membrane/dynamics handover):
- membrane/dynamics-aware distillation:
  - trace pre-ternary “analog membrane” tensors
  - add normalized SmoothL1/MSE alignment losses
  - verify PPL improves vs baseline under same compute

---

## Phase G — v19: Publication & reproducibility

Deliverables:
- reproduction scripts for v15–v18
- paper-ready tables/figures
- a reproducibility checklist

Add-on scripts (tiny end-to-end):
- `scripts/reproduce_posttrain_distill.sh`
- `scripts/reproduce_posttrain_rlvr.sh`  
Both must run small-scale with `--max_steps`, produce artifacts + eval JSON.

---

## 5) New phases from the add‑on (H–K), gated and integrated

## Phase H — Generalist scorecard + data mixture (required before any “generalist scaling”)

Deliverables:
- `eval/suite.yaml`
- `data/mixture.yaml`
- `outputs/eval/eval_baselines.json`

H1. Define the scorecard (small, stable)  
Track at least:
- PPL on fixed LM corpus
- instruction-following pass rate (small fixed set)
- reasoning subset (small)
- coding subset (small executable)
- formatting/safety stability checks (simple verifiers)

H2. Mixture training config  
Explicit weights for:
- general text
- instruction
- code
- reasoning  
Curriculum: ramp context length gradually.

H3. Regression gate  
Any specialist improvement must not regress generalist metrics beyond a defined threshold.

Implementation note:
If useful, integrate EleutherAI lm-evaluation-harness as a backend, but keep your own thin wrapper so results remain stable across harness upgrades.

---

## Phase I — World-fit via retrieval (external memory)

Deliverables:
- `retrieval/` runtime + index tooling
- `outputs/retrieval/retrieval_eval.json`

I1. Two retrieval modes  
- Offline pack: curated corpus + ANN index (mobile-friendly)
- Online mode: optional later

I2. Train “answer with citations” behavior  
- SFT templates that require citations
- citation verifier (string/fuzzy match) that checks cited spans exist in retrieved passages

I3. Retrieval gates  
- citation correctness improves vs baseline
- hallucination rate decreases vs baseline
- latency overhead bounded; caching required

---

## Phase J — Post-training: distillation + RL (capability multiplier)

Deliverables:
- `outputs/posttrain/posttrain_report.md`
- `outputs/posttrain/posttrain_metrics.json`

J1. Strong-to-weak distillation (default path)  
- Off-policy distillation: precompute teacher outputs for a balanced prompt set
- On-policy distillation: student generates, teacher provides token-level targets

J2. RL with verifiable rewards (RLVR) (only where reward is deterministic)  
Targets:
- code: unit tests in sandbox
- math/science: python verifier  
Guardrails:
- KL constraint to reference
- reward-hacking regression suite
- generalist scorecard gate after every RL run  
Optimization:
- Use group-based sampling approaches (GRPO-style) to reduce variance/memory when feasible.

J3. Preference alignment for open-ended tasks (later)  
If/when needed, use DPO-style preference optimization rather than full RLHF.

---

## Phase K — Scaling ladder to 1B (only after gates)

Deliverables:
- `outputs/scale/scale_ladder_report.md`

K1. Steps  
74M → ~300M → ~1B

At each step:
- v15 spike health must remain acceptable (no drift collapse)
- generalist scorecard must improve or remain stable
- v17 benchmarks rerun; efficiency must not collapse

K2. Recipe discipline  
- start from distilled checkpoints
- short SFT stabilization before any RL
- RLVR in short bursts with frequent gates

---

## 6) Mobile trajectory (explicit, because it’s your end goal)

This is not a roadmap reordering; it’s a parallel design constraint.

- Retrieval offline pack must be deployable on-device (small index, cached).
- Define an export format early (weights + tokenizer + metadata).
- Plan for packed low-bit weights (int8/int4/ternary) and an inference runtime that can consume them.
- Keep the core recurrent update simple; it should map to ARM/NEON kernels later.

(Implementation can begin after v17 once correctness + benchmarks exist.)

---

## 7) Risk register (what can kill the project)

- “Spikes are garbage”: v15 fails. Mitigation: fix spike health before sparse work.
- “Sparse is slower”: v16 exists but runs slower. Mitigation: keep it correct; explore structured sparsity; don’t claim wins without v17 data.
- “RL makes it worse”: reward overfitting/hacking. Mitigation: verifiable rewards only + KL + regression gates.
- “Retrieval makes UX bad”: latency spikes. Mitigation: caching + end-to-end benchmarks.
- “Generalist regressions”: specialist tuning breaks basics. Mitigation: scorecard gate.

---

## 8) First integration checklist (merge immediately)

1) Add `eval/` harness skeleton → output one JSON.
2) Add `data/mixture.yaml` placeholder.
3) Add `verifiers/` interface + stubs.
4) Add `retrieval/` interface stub (returns empty).
5) Ensure all of the above are disabled-by-default and covered by tests.

---

## 9) Operations & reporting layer (autonomous execution control)

This section governs day-to-day autonomous execution and reporting. It does **not** change scientific dependencies or reorder v15→v19.

Control docs:
- `docs/ops/AUTONOMY_OPERATING_MODEL.md`
- `docs/ops/STATUS_BOARD.md`
- `docs/ops/GATE_POLICY.md`
- `docs/ops/REPORTING_CONTRACT.md`

Machine-readable state:
- `state/program_status.yaml`
- `state/gate_results.yaml`
- `state/autopilot_queue.yaml`

Reports:
- `reports/index.md`
- `reports/templates/run_report.md`
- `reports/templates/run_report.json`
- `reports/templates/needs_input.md`

Autonomy rules summary:
1) Continue automatically only when required gates are green.
2) Pause automatically and emit `needs_input.md` on any red gate.
3) Yellow gates require mitigation notes in the next run plan.
4) Every run must update:
   - `reports/index.md`
   - `state/program_status.yaml`
   - `state/gate_results.yaml`
   - `docs/ops/STATUS_BOARD.md`

End.
