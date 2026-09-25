# Hilbert

## Pruned & Quantized 4B Decision Model · CUDA Backend · Alloy Governance

**Hilbert** is an experimental pruned 4B semantic-decision model and systems runtime derived from the decision-native architecture demonstrated by SemIf.

The project focuses on a narrow execution path:

```text
state + question + declared options
              │
              ▼
       Hilbert 4B Model
              │
              ▼
       typed option scores
              │
       ┌──────┴──────┐
       ▼             ▼
    CUDA          Governance
   Backend          Layer
       │             │
       ▼             ▼
   Kernels         Alloy
       │          constraints
       └──────┬──────┘
              ▼
        Auditable Result
```

> **Status:** Experimental research implementation.
> Model pruning, distillation, CUDA kernels, and Alloy governance specifications must be independently benchmarked and verified before production claims are made.

---

## 1. What Hilbert Is

Hilbert treats many AI workloads as **typed decisions rather than text-generation problems**.

Instead of asking a model to generate:

```text
"The request should be routed to account-access support because..."
```

Hilbert evaluates declared alternatives directly:

```json
{
  "question": "Which queue should handle this request?",
  "options": [
    {"id": "access", "description": "Account access support."},
    {"id": "billing", "description": "Billing support."}
  ]
}
```

The resulting computation is conceptually:

```text
P(access | state, question, options)
P(billing | state, question, options)
```

This follows the decision-native pattern documented by the upstream SemIf experiment, where declared option logits are read directly without sampling an answer sentence or running a JSON-repair loop.

---

# 2. Hilbert 4B

Hilbert targets a **pruned/distilled 4B-class model**.

The intended transformation is:

```text
Reference semantic model
          │
          ▼
   distillation data
          │
          ▼
   structured decisions
          │
          ▼
       pruning
          │
          ▼
    Hilbert 4B
          │
          ▼
    quantization
          │
          ▼
   CUDA execution
```

The supplied SemIf baseline demonstrates Qwen3.5-4B as a frozen 4B decision model and reports direct-logit evaluation across multiple decision workloads.

Hilbert's pruning/distillation pipeline is a separate experimental layer and should not be interpreted as evidence that the upstream SemIf model itself was distilled or pruned.

---

# 3. Quantization

Hilbert is served **quantized**. The pruned and distilled 4B model is the
thing being measured; quantization is how it fits and runs on consumer GPUs
(the reference target is an RTX 3080, sm_86). A quantized model is judged by
**decision agreement with its BF16 reference**, not by perplexity. A format
that flips decisions is a correctness issue (§13), however much memory it
saves.

### What exists in this repository

Status uses the ladder of §19: PROPOSED → IMPLEMENTED → BENCHMARKED →
INDEPENDENTLY VERIFIED.

| Piece | Where | Status | Evidence |
|---|---|---|---|
| GGUF loading, 13 weight formats (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K; IQ formats refused) | [`cleanroom-transformer/src/gguf.c`](cleanroom-transformer/src/gguf.c), [`ggml_quant.h`](cleanroom-transformer/src/ggml_quant.h) | IMPLEMENTED, tested | Every format dequantizes bit-exactly against llama.cpp's reference. Tiny Llama GGUFs match PyTorch and transformers' GGUF loader within 1e-6 relative ([engine README](cleanroom-transformer/README.md#gguf-files)) |
| Quantized weights resident on the GPU | the same engine, `--gpu-weights quantized` (default) | IMPLEMENTED | Weights stay in GGUF block format and are decoded inside the CUDA kernels: an 8B Q4_K_M needs about 5 GB of weights instead of about 17 GB as BF16. `selftest` checks every quantized-weight kernel against the host decoder. It was reported passing on an RTX 3080; no log is committed |
| Tensor inspection | `cleanroom-transformer dequant --tensor NAME` | IMPLEMENTED | Prints any GGUF tensor as float32 rows, for checking a quantized file by hand |
| Quantized models in the browser | [`webgpu-demo/`](webgpu-demo/README.md) | IMPLEMENTED | Qwen3-0.6B Q8_0, MiniCPM5-2B Q4_K_M and Qwen3.5-4B Q4_K_M, all pinned by revision. The scores the demo shows come from the BF16 checkpoints, not from these quantized files |
| 27B at 5.0 bits per weight (exl3) | [`exl3-bridge/`](exl3-bridge/README.md) | BENCHMARKED | Committed row-level results with SHA256SUMS: `authored144` balanced accuracy 0.9579 (pinned 4B BF16: 0.813); `shape777` argmax agreement 0.8443 with the 4B BF16 predictions (121 flips). Model family and quantization both differ, so this is **not** a quantization ablation |
| Quantized Hilbert 4B | — | PROPOSED | The engine's GGUF loader reads Llama-architecture files only. The Qwen3.5 hybrid layers (Gated DeltaNet and gated attention) need loader and kernel support before a quantized Hilbert 4B can run through the CUDA backend |
| Full 8B GGUF run end to end | — | PROPOSED | Tokenizer and prompts of a released Llama 3 8B Instruct Q4_K_M match Hugging Face; the full forward pass on that file has not been run |

### How a quantized Hilbert 4B will be measured

Each quantization format (for example Q8_0, Q6_K, Q5_K, Q4_K) is compared with
the BF16 Hilbert 4B on the committed fixtures (`shape777`, `authored144`):

```text
BF16 Hilbert 4B ──► row-level decisions ─┐
                                          ├──► decision agreement + flip list
Quantized (format F) ──► row-level ──────┘        balanced accuracy
                                                   calibration error
                                                   memory footprint
                                                   latency (RTX 3080)
```

Record per format: the quantizer and its version, the source checkpoint
revision, the SHA-256 of the quantized file, the kernel revision, and the flip
list itself. Report agreement and accuracy separately (§11): a format can keep
accuracy while changing which rows are right.

---

# 4. Decision-Native Runtime

Hilbert removes unnecessary text-generation work from decision workloads.

### Conventional generation

```text
prompt
  ↓
model
  ↓
tokens
  ↓
JSON/text
  ↓
parser
  ↓
decision
```

### Hilbert

```text
state
  +
question
  +
options
  ↓
model forward pass
  ↓
typed option scores
  ↓
decision
```

The source experiment reports that direct typed logits produced zero output tokens for its measured decision path, while a compact autoregressive JSON baseline generated 111 tokens for the same 21 binary decisions.

---

# 5. CUDA Backend

Hilbert adds a dedicated CUDA execution backend.

```text
Hilbert Runtime
      │
      ▼
CUDA Backend
      │
 ┌────┼────────────────────┐
 ▼    ▼                    ▼
GEMM  Attention            Norm
 │      │                  │
 ▼      ▼                  ▼
Tensor memory        Reduction kernels
      │
      ▼
Typed decision logits
```

The CUDA layer is intended to provide:

* GPU-resident model execution
* fused tensor operations
* reduced intermediate-memory movement
* optimized matrix multiplication
* attention kernels
* normalization kernels
* reduction kernels
* option-score extraction
* deterministic execution modes where supported by the hardware/runtime

CUDA performance numbers should be treated as **unverified until benchmark artifacts are committed**.

---

# 6. CUDA Kernel Boundary

The kernel layer should remain explicit rather than hiding GPU execution behind an opaque abstraction.

Example structure:

```text
cuda/
├── attention/
├── matmul/
├── normalization/
├── reduction/
├── embedding/
├── logits/
├── memory/
└── dispatch/
```

Each kernel should have:

```text
kernel specification
      │
      ├── input dimensions
      ├── output dimensions
      ├── dtype
      ├── memory contract
      ├── synchronization requirements
      ├── numerical tolerance
      └── test vector
```

A kernel is not considered verified merely because it compiles or produces plausible output.

---

# 7. Alloy Governance Layer

Alloy is used as a **relational specification and counterexample engine** around the governance model.

It does not prove the neural network's semantic correctness.

The intended separation is:

```text
Neural computation
       │
       ▼
candidate decision
       │
       ▼
governance specification
       │
       ▼
Alloy analysis
       │
 ┌─────┴─────┐
 ▼           ▼
SAT         UNSAT
 │           │
 ▼           ▼
counter-    constraint
example     boundary
```

Alloy models should specify relationships such as:

```text
Node
Decision
Policy
Authority
Grant
Permission
Resource
Evidence
Revision
```

Example conceptual invariant:

```text
No administrative permission exists
unless an authorized grant relates:

    Node × Grant × Authority × Permission
```

The Alloy model should be used to search for counterexamples to the governance invariants.

---

# 8. Governance Boundary

Hilbert separates **model output** from **authority**.

A model can produce:

```text
decision = ALLOW
```

without possessing authority to authorize the corresponding operation.

The governance layer therefore follows:

```text
MODEL OUTPUT
     │
     ▼
PROPOSED DECISION
     │
     ▼
POLICY EVALUATION
     │
     ▼
AUTHORITY CHECK
     │
     ▼
GOVERNANCE RESULT
```

This prevents:

```text
model confidence
      ≠
administrative authority
```

and:

```text
prediction
      ≠
authorization
```

---

# 9. Auditable Results

Every decision should preserve sufficient metadata to reproduce the computation.

Recommended result structure:

```json
{
  "decision_id": "example-001",
  "model": "hilbert-4b",
  "model_revision": "<PINNED_REVISION>",
  "prompt_sha256": "<HASH>",
  "options": [
    {
      "id": "access",
      "probability": 0.91
    },
    {
      "id": "billing",
      "probability": 0.09
    }
  ],
  "selected": "access",
  "backend": "cuda",
  "kernel_revision": "<PINNED_REVISION>",
  "governance_revision": "<PINNED_REVISION>"
}
```

The upstream experiment similarly records typed scores, timing, model revision, and prompt hashes as part of its audit-oriented result structure.

---

# 10. Reproducibility

Hilbert should pin:

* model revision
* tokenizer revision
* pruning configuration
* distillation dataset revision
* CUDA version
* GPU architecture
* kernel revision
* compiler version
* numerical precision and quantization format (with the quantizer version and the quantized file's SHA-256)
* benchmark fixture
* governance specification revision
* Alloy model revision

A result without its corresponding revision metadata should be treated as incomplete evidence.

---

# 11. Benchmark Structure

```text
benchmarks/
├── decisions/
├── pruning/
├── distillation/
├── cuda/
├── numerical/
├── governance/
├── alloy/
└── reproduction/
```

Each benchmark should distinguish:

```text
correctness
performance
numerical agreement
decision agreement
governance validity
```

These properties must not be collapsed into one score.

---

# 12. Model Quality

The upstream SemIf experiment provides a useful reference methodology: fixed workloads, frozen models, explicit metrics, prompt hashes, timing measurements, and committed raw results.

Hilbert should therefore report at minimum:

| Metric                 | Purpose                                 |
| ---------------------- | --------------------------------------- |
| Decision accuracy      | Semantic decision quality               |
| Balanced accuracy      | Class-balanced evaluation               |
| Calibration error      | Confidence reliability                  |
| Decision agreement     | Comparison against reference model      |
| Perturbation stability | Sensitivity to controlled input changes |
| Kernel numerical error | CUDA correctness                        |
| Throughput             | Runtime performance                     |
| Latency                | Per-decision response                   |
| Memory footprint       | Deployment requirements                 |
| Governance violations  | Policy-model boundary testing           |

---

# 13. CUDA Correctness

Performance cannot substitute for numerical verification.

For each optimized kernel:

```text
Reference implementation
          │
          ▼
High-precision / trusted baseline
          │
          ▼
CUDA implementation
          │
          ▼
comparison
```

Record:

```text
absolute error
relative error
maximum error
mean error
ULP difference where applicable
NaN/Inf behavior
edge-case behavior
```

A faster kernel that changes the decision boundary must be treated as a correctness issue, not merely a performance optimization.

---

# 14. Alloy Counterexample Testing

Governance specifications should deliberately attempt to break themselves.

Example challenge classes:

```text
duplicate authority
revoked grant
expired grant
conflicting policy
orphaned node
unauthorized node
circular authority
missing evidence
stale revision
conflicting permissions
```

The expected workflow is:

```text
Invariant
   ↓
Alloy model
   ↓
bounded analysis
   ↓
counterexample
   ↓
repair specification
   ↓
rerun analysis
```

An Alloy model finding no counterexample within a finite scope is **not equivalent to an unrestricted mathematical proof**.

---

# 15. Negative-Gate Principle

Hilbert adopts an explicit negative gate:

```text
UNVERIFIED
    ↓
DO NOT PROMOTE TO VERIFIED
```

Therefore:

```text
compiles
    ≠
correct

passes test
    ≠
proven

high confidence
    ≠
authority

CUDA speedup
    ≠
semantic correctness

Alloy SAT result
    ≠
system correctness

absence of a bounded counterexample
    ≠
universal proof
```

This distinction is central to the project.

---

# 16. Repository Composition

Current repository composition:

```text
Python       44.1%
C            43.0%
CUDA          6.5%
JavaScript    3.5%
HTML          1.7%
Alloy         0.8%
Makefile      0.4%
```

These shares predate [`playground/`](playground/README.md) (TypeScript, C
compiled to WebAssembly, Swift) and [`roaming/`](roaming/README.md) (C#);
GitHub's language bar has the current figures.

The stack deliberately keeps the CUDA and Alloy layers visible in the repository rather than hiding them behind a single runtime abstraction.

---

# 17. Directory Layout

What exists today:

| Path | What it is |
|---|---|
| [`cleanroom-transformer/`](cleanroom-transformer/README.md) | The C/CUDA decision engine: Qwen3.5 and Llama 3 forward passes, GGUF quantized weights, sm_86 kernels (`ep/` holds the Equilibrium Propagation GEMM kernels), Alloy shape proofs (`formal/`), the decision memory and its ZK circuit (`zk/`) |
| [`playground/`](playground/README.md) | The decision-memory commands in the browser (Cloudscape UI, freestanding WebAssembly core) and a Swift host |
| [`roaming/`](roaming/README.md) | `roam`: portable `.roam` session bundles, audited God mode, Windows GodMode folder, in Node.js and C# |
| [`webgpu-demo/`](webgpu-demo/README.md) | Browser-only inference with quantized GGUF models |
| [`exl3-bridge/`](exl3-bridge/README.md) | The exl3 (exllamav3) quantized readout track |
| `src/`, `benchmarks/`, `results/`, `docs/`, `demo/` | The SemIf reference implementation, fixtures, committed results, method and results documents, and the replay demo |

The layout Hilbert is growing toward:

```text
hilbert/
├── model/
│   ├── config/
│   ├── tokenizer/
│   ├── pruning/
│   └── distillation/
│
├── runtime/
│   ├── python/
│   └── c/
│
├── cuda/
│   ├── attention/
│   ├── matmul/
│   ├── normalization/
│   ├── reduction/
│   ├── logits/
│   └── dispatch/
│
├── alloy/
│   ├── governance/
│   ├── authority/
│   ├── permissions/
│   └── counterexamples/
│
├── benchmarks/
│   ├── decisions/
│   ├── cuda/
│   ├── model/
│   └── governance/
│
├── web/
│   ├── demo/
│   └── assets/
│
├── docs/
│   ├── METHOD.md
│   ├── RESULTS.md
│   ├── REPRODUCE.md
│   ├── CUDA.md
│   └── GOVERNANCE.md
│
├── Makefile
└── README.md
```

---

# 18. Relationship to SemIf

Hilbert is inspired by the architecture documented in the supplied SemIf project:

* runtime-defined criteria
* decision-native scoring
* direct option-logit evaluation
* shared-state execution
* reproducible fixtures
* row-level results
* pinned model revisions
* prompt hashing
* explicit claim boundaries

These characteristics are documented in the supplied project material.

Hilbert adds its own experimental layers:

```text
SemIf decision architecture
            │
            ▼
       Hilbert 4B
            │
      ┌─────┴─────┐
      ▼           ▼
   CUDA         Alloy
   backend      governance
      │           │
      └─────┬─────┘
            ▼
      Hilbert Runtime
```

Hilbert should not represent upstream SemIf measurements as measurements of Hilbert.

---

# 19. Claim Discipline

The project distinguishes four states:

```text
PROPOSED
    ↓
IMPLEMENTED
    ↓
BENCHMARKED
    ↓
INDEPENDENTLY VERIFIED
```

A feature remains **PROPOSED** until implementation evidence exists.

A feature remains **IMPLEMENTED** until benchmark evidence exists.

A benchmark result remains **BENCHMARKED** until its reproduction procedure is independently confirmed.

This prevents README claims from becoming stronger than the underlying evidence.

---

# 20. License

This repository should include an explicit root license and identify the license applicable to each third-party dependency and model.

Model weights and upstream components retain their respective licenses unless their licenses explicitly permit redistribution.

The supplied SemIf project, for comparison, states that its project code is MIT licensed while upstream model weights retain their own licenses.

---

# 21. Research Status

Hilbert is an experimental systems research project combining:

```text
4B model compression
+
quantization
+
decision-native inference
+
CUDA kernel engineering
+
GPU execution
+
formal governance modeling
+
Alloy counterexample analysis
+
reproducible benchmarking
```

The central engineering question is:

> **How small can a decision-oriented model become while retaining measurable decision quality, efficient GPU execution, reproducibility, and an explicit machine-checkable governance boundary?**

The answer must come from the benchmarks, kernel tests, model evaluations, and Alloy counterexamples—not from the README.

---

## Hilbert

```text
PRUNE THE MODEL.
EXECUTE THE DECISION.
VERIFY THE BOUNDARY.
```
