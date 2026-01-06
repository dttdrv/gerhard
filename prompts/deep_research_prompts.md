# Deep Research Prompts: ASNN-Goose Knowledge Distillation

## Research Objective

Gather comprehensive, implementation-ready knowledge for distilling GPT-2 (transformer, 124M params) into a Spiking Neural Network (SNN) student model (~50M params) using advanced distillation techniques.

---

## Prompt 1: Learnable Temperature in Knowledge Distillation (CTKD)

### Query

```
Provide a comprehensive technical analysis of learnable temperature in knowledge distillation, specifically the CTKD (Curriculum Temperature for Knowledge Distillation) approach from ArXiv 2211.16231.

REQUIRED SECTIONS:

1. THEORETICAL FOUNDATION
   - Why does temperature matter in KL divergence for distillation?
   - Mathematical derivation: How does T affect gradient magnitude? (Show the T² correction factor derivation)
   - What is the "dark knowledge" hypothesis and how does temperature expose it?
   - Optimal temperature range: Why do papers typically use T ∈ [1, 20]?

2. CTKD SPECIFIC IMPLEMENTATION
   - Log-parameterization: Why use log(T) instead of T directly?
   - Gradient flow through learnable temperature
   - Separate learning rate for temperature (typical values: 0.001-0.1)
   - Initialization strategies (T_init = 1, 2, 4, or 20?)

3. TEMPERATURE DYNAMICS DURING TRAINING
   - Expected temperature trajectory (does it increase or decrease?)
   - Correlation between temperature and training stage
   - When temperature gets stuck at boundaries (1.0 or max clamp)
   - Temperature oscillation and instability patterns

4. REGULARIZATION STRATEGIES
   - Why does unregularized temperature run away to maximum?
   - L2 regularization towards anchor: λ(T - T_target)²
   - Entropy-based regularization alternatives
   - Curriculum-based temperature targets (high→low over training)

5. IMPLEMENTATION PITFALLS
   - Temperature runaway: causes, detection, prevention
   - Gradient explosion when T→0
   - Interaction with mixed precision training (FP16 underflow)
   - Batch size sensitivity

6. EMPIRICAL RESULTS
   - CTKD improvements over fixed temperature (cite specific numbers)
   - Comparison: learnable T vs grid search for optimal fixed T
   - Task-specific optimal temperature ranges (vision vs NLP)

7. CODE PATTERNS
   - PyTorch implementation of LearnableTemperature module
   - Optimizer setup with separate param groups
   - Logging and monitoring temperature during training

Please cite specific papers, provide exact equations with variable definitions, and include PyTorch code snippets where applicable.
```

---

## Prompt 2: Progressive/Curriculum Knowledge Distillation (POCL)

### Query

```
Provide exhaustive technical details on Progressive and Curriculum Learning approaches to Knowledge Distillation, with focus on the POCL framework (ArXiv 2506.05695) and related multi-stage training strategies.

REQUIRED SECTIONS:

1. CURRICULUM LEARNING THEORY
   - Bengio et al.'s curriculum learning foundations
   - Self-paced learning vs predetermined curriculum
   - Sample difficulty metrics for distillation (loss-based, confidence-based)
   - Curriculum scheduling functions (linear, exponential, step)

2. PROGRESSIVE DISTILLATION STAGES
   - Stage 1 (Early): Soft targets with high temperature (T=4-20)
     * Why start soft? (student capacity, gradient smoothness)
     * Alignment weight scheduling (0% → gradual increase)
   - Stage 2 (Middle): Medium temperature (T=2-4)
     * Transitioning from soft to hard targets
     * Introducing auxiliary losses (hidden alignment, attention)
   - Stage 3 (Late): Hard targets (T=1-2)
     * Fine-tuning on true distribution
     * When to freeze certain components

3. STAGE TRANSITION MECHANISMS
   - Time-based transitions (fixed percentages: 40%, 70%, 100%)
   - Loss-based transitions (move to next stage when loss plateaus)
   - Validation-based transitions (when val PPL stops improving)
   - Smooth vs hard transitions (linear interpolation vs step)

4. LOSS WEIGHT SCHEDULING
   - KL loss weight over stages
   - Hidden alignment weight scheduling
   - Cross-entropy (hard label) weight scheduling
   - Total loss = α(t)L_KL + β(t)L_align + γ(t)L_CE

5. TEMPERATURE SCHEDULING (NON-LEARNABLE)
   - Linear annealing: T(t) = T_max - (T_max - T_min) × (t/T_total)
   - Cosine annealing: T(t) = T_min + 0.5(T_max - T_min)(1 + cos(πt/T_total))
   - Step decay: T_stages = [4.0, 2.5, 1.5]

6. INTERACTION WITH LEARNABLE TEMPERATURE
   - Using stage-based targets as regularization anchors
   - Letting temperature learn within stage constraints
   - Curriculum for temperature learning rate

7. IMPLEMENTATION DETAILS
   - get_stage_params(step, total_steps) function design
   - Logging stage transitions
   - Handling stage transitions with learning rate warmup
   - Gradient accumulation across stage boundaries

8. EMPIRICAL EVIDENCE
   - Ablation: progressive vs constant temperature
   - Optimal stage ratios (40/30/30 vs 33/33/33 vs 50/30/20)
   - Task-specific stage configurations

Provide mathematical formulations, training curves showing stage effects, and complete PyTorch implementation patterns.
```

---

## Prompt 3: Hidden State Alignment in Knowledge Distillation

### Query

```
Provide comprehensive technical analysis of hidden state (intermediate layer) alignment for knowledge distillation, focusing on TinyBERT (ArXiv 1909.10351), MiniLM, and adaptations for heterogeneous architectures.

REQUIRED SECTIONS:

1. THEORETICAL MOTIVATION
   - Why align hidden states, not just output logits?
   - Information bottleneck perspective
   - Layer-wise knowledge transfer theory
   - Dark knowledge in intermediate representations

2. LAYER MAPPING STRATEGIES
   - Uniform mapping: student_layer[i] ↔ teacher_layer[i × (T_layers/S_layers)]
   - Skip mapping: align every Nth teacher layer
   - Learned mapping: attention-based layer selection
   - Last-K layers only (often most effective)

3. DIMENSION MISMATCH HANDLING
   - Linear projection: W_proj ∈ ℝ^(d_teacher × d_student)
   - MLP projection with nonlinearity
   - Factorized projection for efficiency
   - Should projector be frozen or learned?

4. ALIGNMENT LOSS FUNCTIONS
   - MSE loss: ||proj(h_s) - h_t||²
   - Cosine similarity: 1 - cos(proj(h_s), h_t)
   - CKA (Centered Kernel Alignment)
   - Attention transfer: ||A_s - A_t||² (for attention maps)

5. LOSS WEIGHTING STRATEGIES
   - Fixed weight (typical: 0.001 - 0.1 of total loss)
   - Per-layer weights (later layers weighted higher)
   - Adaptive weighting based on layer difficulty
   - Curriculum: introduce alignment gradually

6. ARCHITECTURE-SPECIFIC CONSIDERATIONS
   - Transformer → Transformer: attention pattern alignment
   - Transformer → RNN/LSTM: temporal hidden state mapping
   - Transformer → SNN: membrane potential alignment challenges
   - Handling different sequence lengths

7. FAILURE MODES
   - Alignment loss dominating KL loss (weight too high)
   - Gradient conflict between alignment and KL objectives
   - Projector overfitting to teacher activations
   - Layer mismatch causing negative transfer

8. IMPLEMENTATION PATTERNS
   ```python
   def compute_hidden_alignment_loss(t_hiddens, s_hiddens, projector,
                                      teacher_layers, student_layers):
       # Implementation details needed
   ```
   - Efficient hidden state extraction
   - Memory-efficient computation (gradient checkpointing)
   - Handling variable-length sequences

9. EMPIRICAL GUIDELINES
   - TinyBERT: alignment weight = 1.0 (too high for some tasks)
   - DistilBERT: no hidden alignment (logits only)
   - MiniLM: attention alignment only
   - Recommended starting point: weight = 0.01, increase if stable

Provide layer mapping diagrams, loss curves showing alignment effects, and production-ready code.
```

---

## Prompt 4: Spiking Neural Networks for Language Modeling

### Query

```
Provide exhaustive technical analysis of Spiking Neural Networks (SNNs) applied to language modeling and NLP tasks, with focus on knowledge distillation from transformers.

REQUIRED SECTIONS:

1. SNN FUNDAMENTALS FOR NLP
   - Leaky Integrate-and-Fire (LIF) neurons for sequence processing
   - Membrane potential dynamics: V(t+1) = βV(t) + I(t) - S(t)V_th
   - Spike generation: S(t) = Θ(V(t) - V_th)
   - Temporal coding vs rate coding for language

2. SURROGATE GRADIENT METHODS
   - Why SNNs are non-differentiable (Heaviside step function)
   - Surrogate gradient functions:
     * Straight-Through Estimator (STE)
     * Sigmoid surrogate: σ'(x) = σ(x)(1-σ(x))
     * Triangular surrogate
     * SuperSpike: 1/(1 + k|x|)²
   - Gradient scaling and the "dying neuron" problem

3. TERNARY SPIKE REPRESENTATIONS
   - Binary spikes: {0, 1}
   - Ternary spikes: {-1, 0, +1}
   - Why ternary? (bidirectional information flow)
   - Trainable thresholds and amplitudes

4. SNN ARCHITECTURES FOR TRANSFORMERS
   - Spiking self-attention mechanisms
   - Spike-based key-query-value computation
   - Temporal attention patterns
   - Spiking FFN alternatives

5. RECURRENT SNN DESIGNS (GOOSE-STYLE)
   - State-space model inspiration (S4, Mamba)
   - Recurrent processing of transformer-like representations
   - Parallel scan for efficient training
   - Memory and gating with spikes

6. KNOWLEDGE DISTILLATION TO SNNs
   - Challenges: continuous teacher → discrete student
   - Soft label generation for spike outputs
   - Temperature scaling for SNN logits
   - Alignment of membrane potentials to hidden states

7. QUANTIZATION AND EFFICIENCY
   - Binary/ternary weight quantization
   - Activation quantization (spike counts)
   - Energy estimation: MAC vs AC operations
   - Neuromorphic hardware mapping (Loihi, SpiNNaker)

8. TRAINING CONSIDERATIONS
   - Batch normalization alternatives (Threshold-dependent BN)
   - Learning rate schedules for SNNs
   - Gradient clipping requirements
   - Time step selection (T=4, 8, 16?)

9. EVALUATION METRICS
   - Perplexity (standard)
   - Spike sparsity: fraction of zero activations
   - Energy efficiency: theoretical vs measured
   - Latency in time steps

10. STATE-OF-THE-ART RESULTS
    - SpikeBERT, SpikingBERT, SpikeGPT
    - Best reported perplexity on PTB, WikiText
    - Efficiency gains over transformers

Provide membrane potential equations, spike timing diagrams, and PyTorch code for spiking layers.
```

---

## Prompt 5: Ternary Quantization with Learnable Parameters

### Query

```
Provide comprehensive technical details on ternary quantization for neural networks, focusing on channel-wise approaches like TTQ (Trained Ternary Quantization) and TerViT.

REQUIRED SECTIONS:

1. TERNARY QUANTIZATION BASICS
   - Ternary representation: {-α, 0, +α}
   - Threshold-based assignment:
     * q(x) = +α if x > Δ
     * q(x) = 0 if |x| ≤ Δ
     * q(x) = -α if x < -Δ
   - Comparison to binary quantization

2. LEARNABLE PARAMETERS
   - Threshold (Δ): determines sparsity
   - Amplitude (α): determines scale
   - Per-tensor vs per-channel vs per-group
   - Initialization strategies

3. TTQ (TRAINED TERNARY QUANTIZATION)
   - Original paper methodology
   - Separate α+ and α- (asymmetric)
   - STE (Straight-Through Estimator) for gradients
   - Threshold learning dynamics

4. CHANNEL-WISE TERNARY
   - Why per-channel? (different feature scales)
   - α ∈ ℝ^C, Δ ∈ ℝ^C for C channels
   - Memory overhead vs accuracy tradeoff
   - Gradient flow through channel parameters

5. DYNAMIC THRESHOLD COMPUTATION
   - Statistics-based: Δ = α × mean(|x|)
   - Learned: Δ as trainable parameter
   - Adaptive: Δ based on activation distribution
   - Clamping strategies (min=0.01, max=10.0)

6. REGULARIZATION
   - Amplitude regularization: λ × Var(α) across channels
   - Threshold regularization: prevent collapse to 0 or ∞
   - Sparsity regularization: encourage zeros
   - L1/L2 on ternary parameters

7. TRAINING STRATEGIES
   - Gradual quantization (start full precision, anneal)
   - Quantization-aware training from scratch
   - Knowledge distillation with ternary student
   - Mixed precision: which layers to quantize?

8. IMPLEMENTATION
   ```python
   class ChannelWiseTernarySpike(nn.Module):
       def __init__(self, d_model, alpha_init=1.0):
           # Full implementation needed
   ```
   - Forward pass with STE
   - Backward pass gradient computation
   - Parameter initialization
   - Inference optimization

9. INTEGRATION WITH SNNs
   - Ternary spikes vs ternary weights
   - Combining threshold learning with spike generation
   - Membrane potential quantization
   - Energy benefits of ternary operations

10. EMPIRICAL RESULTS
    - Accuracy drop: ternary vs full precision
    - Compression ratios achieved
    - Speedup on CPU/GPU/NPU
    - Best practices from literature

Include mathematical derivations, gradient flow diagrams, and optimized PyTorch implementations.
```

---

## Prompt 6: PyTorch Optimization for Training Speed

### Query

```
Provide exhaustive technical guide on PyTorch training optimization techniques, focusing on knowledge distillation workloads with mixed precision and gradient accumulation.

REQUIRED SECTIONS:

1. MIXED PRECISION TRAINING
   - torch.cuda.amp.autocast() internals
   - GradScaler: why and how it works
   - Ops that run in FP16 vs FP32
   - Loss scaling strategies (dynamic vs fixed)
   - Common pitfalls: NaN gradients, underflow

2. TORCH.COMPILE (PYTORCH 2.0+)
   - Compilation modes: default, reduce-overhead, max-autotune
   - TorchDynamo and TorchInductor
   - What gets compiled? (graphs, not eager ops)
   - Warmup time and when to use
   - Fallback handling for unsupported ops
   - Memory implications

3. TORCH.INFERENCE_MODE VS TORCH.NO_GRAD
   - Differences in implementation
   - Performance implications
   - When to use which
   - Nested context behavior

4. DATA LOADING OPTIMIZATION
   - num_workers tuning (CPU cores vs GPU utilization)
   - pin_memory=True and non_blocking=True
   - Prefetch factor
   - Persistent workers
   - Custom collate functions

5. MEMORY OPTIMIZATION
   - Gradient checkpointing (torch.utils.checkpoint)
   - Activation memory vs parameter memory
   - In-place operations carefully
   - Empty cache strategically
   - Memory-efficient attention (xformers, flash attention)

6. OPTIMIZER EFFICIENCY
   - Fused optimizers: AdamW(fused=True)
   - Foreach implementations
   - Gradient clipping efficiency
   - Learning rate scheduling overhead

7. GRADIENT ACCUMULATION
   - Correct loss scaling: loss / accumulation_steps
   - When to sync gradients (DDP)
   - Memory profile with accumulation
   - Effective batch size calculation

8. EVALUATION EFFICIENCY
   - eval_interval tuning
   - Subset validation for speed
   - Caching teacher outputs
   - Non-blocking data transfers

9. PROFILING AND DEBUGGING
   - torch.profiler usage
   - CUDA event timing
   - Memory snapshots
   - Bottleneck identification

10. DISTRIBUTED TRAINING
    - DDP vs FSDP for distillation
    - Gradient synchronization
    - Teacher model placement
    - Communication overlap

11. SPECIFIC TO DISTILLATION
    - Teacher in eval mode: .eval() and no_grad
    - Caching teacher logits
    - Async teacher forward
    - KL divergence computation efficiency

Provide benchmark numbers, profiling examples, and production-ready code patterns.
```

---

## Prompt 7: Validation and Testing for ML Training Pipelines

### Query

```
Provide comprehensive guide on validation, testing, and quality assurance for machine learning training pipelines, specifically for knowledge distillation experiments.

REQUIRED SECTIONS:

1. TRAINING SANITY CHECKS
   - Loss decreasing check (first N steps)
   - Gradient norm monitoring (detect explosion/vanishing)
   - Learning rate warmup verification
   - Parameter update magnitude tracking

2. DISTILLATION-SPECIFIC VALIDATIONS
   - Temperature evolution (not stuck at boundaries)
   - KL divergence reasonable range
   - Student-teacher output correlation
   - Alignment loss decreasing (if applicable)

3. NUMERICAL STABILITY TESTS
   - NaN/Inf detection in loss and gradients
   - Mixed precision underflow detection
   - Gradient clipping activation frequency
   - Loss scale history (GradScaler)

4. MODEL QUALITY METRICS
   - Perplexity trajectory (should decrease)
   - Validation vs training loss gap
   - Teacher-student output distribution similarity
   - Spike statistics for SNNs

5. PERFORMANCE REGRESSION TESTS
   - Baseline comparisons (vs previous version)
   - Wall-clock time per step
   - Throughput (tokens/second)
   - Memory usage peaks

6. AUTOMATED TEST DESIGN
   ```python
   def test_training_pipeline():
       # What to test?
       # How to make tests fast?
       # Integration vs unit tests?
   ```

7. CHECKPOINT VALIDATION
   - Model loadable after save
   - Optimizer state preserved
   - Training resumable
   - Reproducibility checks

8. LOGGING BEST PRACTICES
   - What to log at each step
   - Logging frequency tradeoffs
   - Structured logging for analysis
   - Real-time monitoring (TensorBoard, W&B)

9. FAILURE DETECTION AND RECOVERY
   - Automatic NaN detection and recovery
   - Checkpoint rollback on failure
   - Graceful degradation
   - Alert mechanisms

10. POST-TRAINING VALIDATION
    - Final perplexity within expected range
    - All N tests pass framework
    - Model size verification
    - Export and inference testing

Provide pytest patterns, logging schemas, and monitoring dashboards.
```

---

## Usage Instructions

1. **Run each prompt separately** for focused, deep responses
2. **Follow up** on specific sections that need more detail
3. **Request code examples** for any theoretical concept
4. **Ask for citations** to verify claims with original papers
5. **Combine findings** into your implementation

## Priority Order

1. **Prompt 1 (Learnable Temperature)** - Critical for current v11.1 bug fix
2. **Prompt 6 (PyTorch Optimization)** - For speed improvements
3. **Prompt 2 (Progressive Distillation)** - For future v11.2+ work
4. **Prompt 4 (SNNs for NLP)** - Core architecture understanding
5. **Prompt 3 (Hidden Alignment)** - For future technique addition
6. **Prompt 5 (Ternary Quantization)** - For channel-wise spikes
7. **Prompt 7 (Validation)** - For robust testing
