# LUMOS OSS Review Report & Fix Plan

> **Created**: 2026-06-26
> **Status**: Pending fixes — nothing committed to the public repo yet
> **Public Repo**: [github.com/dhruvnigam93/lumos-user-model](https://github.com/dhruvnigam93/lumos-user-model)
> **Internal Repo**: `dream11/ds-giveaways-lumos` (private)

---

## Background Context (Read This First)

### What is this project?

LUMOS (Large User MOdel Series) is a cross-attention encoder-decoder transformer built at Dream11 for multi-task user behavior prediction. It was published as [arXiv:2512.08957](https://arxiv.org/abs/2512.08957). We are open-sourcing it into a standalone public repo (`dhruvnigam93/lumos-user-model`) completely disconnected from Dream11's internal infrastructure.

### What has been done so far?

1. A public GitHub repo was created at `dhruvnigam93/lumos-user-model` (currently shows only an auto-generated LICENSE file — nothing else has been committed yet).
2. A local project was scaffolded at `/Users/dhruvnigam/Projects/lumos-oss/` with all code written but **not yet committed or pushed**.
3. The code was extracted from the internal repo (`/Users/dhruvnigam/Projects/ds-giveaways-lumos/`) and genericized:
   - `CrossAttentionForecaster` → `LUMOS`
   - `supply_*` → `event_context_*`
   - `user_embed` → `activity_embed`
   - All Databricks/MLflow/PySpark/Delta dependencies removed
   - All Dream11-specific names, emails, credentials, S3 paths stripped
4. An adversarial staff-engineer-level review was run against the implementation. It found **49 issues** (5 Critical, 10 High, 19 Medium, 15 Low). **Zero proprietary leaks were found** — the code is clean.

### Project structure (as written, not yet committed)

```
lumos-oss/
├── .github/workflows/ci.yml     # GitHub Actions CI
├── .gitignore
├── LICENSE                       # Apache 2.0 (has placeholder issue — see C2)
├── README.md
├── pyproject.toml
├── lumos/
│   ├── __init__.py
│   ├── config.py                 # LUMOSConfig, TrainingConfig, TaskConfig dataclasses
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py          # MultiHeadAttention
│   │   ├── layers.py             # EncoderLayer, DecoderLayer, TransformerEncoder, TransformerDecoder
│   │   ├── embeddings.py         # ConfigurableEmbedding (MLP-based input embedding)
│   │   ├── positional.py         # 5 positional embedding types + factory
│   │   └── lumos.py              # LUMOS main model class
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py             # MultiTaskLoss, DirectLearnedWeightedLoss
│   │   ├── schedulers.py         # LR scheduler factory
│   │   ├── monitor.py            # TrainingMonitor (TensorBoard + system metrics)
│   │   └── trainer.py            # LUMOSTrainer (single-GPU training loop)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py          # Synthetic dataset generator (5 user archetypes)
│   │   ├── dataset.py            # UserSequenceDataset (PyTorch Dataset)
│   │   └── preprocessing.py      # MinMaxScaler, LogMinMaxScaler, build_sequences
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── extractor.py          # UserEmbeddingExtractor (batch extraction, similarity)
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py            # evaluate_binary, evaluate_continuous, compute_ece
├── tests/
│   ├── __init__.py
│   ├── test_model.py             # 21 tests (attention, layers, positional, LUMOS)
│   ├── test_losses.py            # 5 tests (MultiTaskLoss, DirectLearnedWeightedLoss)
│   ├── test_data.py              # 10 tests (synthetic, dataset, scalers)
│   └── test_embeddings.py        # 3 tests (extraction, similarity)
└── examples/
    ├── quickstart.py
    ├── train_synthetic.py
    └── extract_embeddings.py
```

### Original source files the code was extracted from

| Extracted module | Original internal file |
|-----------------|----------------------|
| `lumos/model/lumos.py` | `src/dn/models/model_cross_attn.py` (`CrossAttentionForecaster`) |
| `lumos/model/positional.py` | `src/dn/models/pos_embeddings.py` |
| `lumos/training/losses.py` | `src/dn/utils/domain/training_ddp.py` (`MultiTaskLoss`, `DirectLearnedWeightedLoss`) |
| `lumos/training/trainer.py` | `src/dn/utils/domain/training_ddp.py` (`train_ddp()`) |
| `lumos/training/monitor.py` | `src/dn/utils/domain/train_monitor.py` |
| `lumos/evaluation/metrics.py` | `src/dn/eval/PredictionEvaluator.py` + `src/dn/utils/service/mlutils.py` |
| `lumos/config.py` | `src/dn/entities.py` (genericized) |
| `lumos/data/*` | New code (no internal equivalent — internal used PySpark/Databricks/Delta) |
| `lumos/embeddings/*` | New code (internal equivalent was a Databricks notebook) |

### What verifications passed

- `LUMOS.from_paper_config()` instantiates successfully (14.69M params — see C3 about mismatch)
- Forward pass produces correct output shapes (`[B, n_targets]` for aggregated, `[B, T, n_targets]` for sequence)
- `historical_user_embedding()` works with all 4 reduction strategies (mean, max, expavg, last)
- `MultiTaskLoss` and `DirectLearnedWeightedLoss` compute and backpropagate correctly
- Synthetic data generates, `UserSequenceDataset` loads from it, full training loop runs end-to-end
- `UserEmbeddingExtractor` extracts embeddings and computes similarities
- `evaluate_binary()` and `evaluate_continuous()` return correct metric dictionaries
- **41 tests pass** in pytest
- **Zero proprietary references** found in full grep sweep
- Architectural fidelity verified: core model is a faithful port of the original

---

## Complete Findings

### Severity Legend

- **CRITICAL** — Must fix before any commit to the public repo. These are showstoppers.
- **HIGH** — Should fix before announcing/sharing the repo publicly. Significant quality or completeness gaps.
- **MEDIUM** — Fix when possible. Code quality, API design, dead code.
- **LOW** — Nitpicks. Fix if touching the relevant file anyway.

---

## CRITICAL (5)

### C1. Nothing has been committed to the public repo

**Location**: Entire repo — `git status` shows all files as untracked
**What's wrong**: The public repo at `github.com/dhruvnigam93/lumos-user-model` contains only the auto-generated LICENSE file from `gh repo create`. All source code, tests, examples, CI config, and README are sitting as untracked files on disk at `/Users/dhruvnigam/Projects/lumos-oss/`. The original plan specified "commit & push" at the end of each step (Steps 1–11). None of those commits were made.
**What it should be**: All code committed and pushed in logical commits (one per step, or at minimum one clean initial commit).
**Why it matters**: Anyone visiting the public repo right now sees an empty project. CI cannot run. The repo is unusable.

**Fix**:
```bash
cd /Users/dhruvnigam/Projects/lumos-oss

# Clean up build artifacts first (see C4)
rm -rf .pytest_cache/ lumos_user_model.egg-info/

# Stage all source files (NOT uv.lock — see C4)
git add .gitignore pyproject.toml README.md LICENSE \
       lumos/ tests/ examples/ .github/

git commit -m "feat: initial LUMOS open-source release

Cross-attention encoder-decoder transformer for multi-task user behavior
prediction. Extracted and genericized from internal implementation.

Includes: model architecture, training loop, synthetic data generator,
embedding extractor, evaluation metrics, tests, examples, and CI."

git push origin main
```

---

### C2. LICENSE copyright line is a template placeholder

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/LICENSE`, appendix section near bottom
**What's wrong**: The copyright notice reads `Copyright [yyyy] [name of copyright owner]` — the template placeholders were never filled in when `gh repo create --license apache-2.0` generated the file.
**What it should be**: `Copyright 2024 Dhruv Nigam` (or whichever year/entity is appropriate — check if Dream11 needs to be credited or if this is a personal project).
**Why it matters**: An Apache 2.0 license without a proper copyright notice is legally ambiguous. This is a hard requirement for any public repo.

**Fix**: Open `LICENSE`, find the `[yyyy] [name of copyright owner]` line, and replace with the correct year and name.

---

### C3. `from_paper_config()` parameter count is 2x what the plan claims

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/lumos/model/lumos.py`, lines 97–119
**What's wrong**: `LUMOS.from_paper_config()` produces **14,691,845 parameters (14.69M)**. The original plan (Step 2 verification) claimed it should produce **~7.23M**. This is a 2x discrepancy.
**What it should be**: Whichever number the actual paper (arXiv:2512.08957) reports. Either the `from_paper_config()` hyperparameters are wrong, or the plan's estimate was wrong.
**Why it matters**: If the paper reports 7.23M params, then `from_paper_config()` produces the wrong model and anyone trying to reproduce the paper results will get incorrect results. This directly undermines the "from paper" claim.

**Fix**:
1. Open the paper at https://arxiv.org/abs/2512.08957
2. Find the reported parameter count and model dimensions
3. Adjust the defaults in `from_paper_config()` to match exactly
4. Update the verification test in `tests/test_model.py` to assert the correct param count

---

### C4. Build artifacts will be committed if `git add -A` is used

**Location**: `.gitignore` (missing entries) + filesystem
**What's wrong**: Three categories of build artifacts exist on disk and would be committed:
- `/Users/dhruvnigam/Projects/lumos-oss/.pytest_cache/` — **not in `.gitignore`**
- `/Users/dhruvnigam/Projects/lumos-oss/lumos_user_model.egg-info/` — covered by `*.egg-info/` in `.gitignore` but exists on disk
- `/Users/dhruvnigam/Projects/lumos-oss/uv.lock` — **not in `.gitignore`**
**What it should be**: Add `.pytest_cache/` and `uv.lock` to `.gitignore`. Delete the artifact directories before committing.

**Fix**: Add these lines to `.gitignore`:
```
.pytest_cache/
uv.lock
```
Then clean up:
```bash
rm -rf .pytest_cache/ lumos_user_model.egg-info/
```

---

### C5. `psutil` import crashes on base install

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/lumos/training/monitor.py`, line 1
**What's wrong**: `import psutil` is at module top level. The import chain is:
1. `lumos/training/__init__.py` imports `LUMOSTrainer` from `trainer.py`
2. `trainer.py` imports `TrainingMonitor` from `monitor.py`
3. `monitor.py` does `import psutil` unconditionally

But `psutil` is only listed in the `[training]` optional extra in `pyproject.toml`, not in base dependencies. So anyone who installs with plain `pip install lumos-user-model` (no extras) and runs `from lumos.training import LUMOSTrainer` gets:
```
ImportError: No module named 'psutil'
```
**What it should be**: Either (a) make `psutil` a core dependency, or (b) guard the import:
```python
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
```
And handle the fallback in `get_system_metrics()`.
**Why it matters**: This breaks the README's training example on a base install. First impression for new users will be a crash.

**Fix** (option b — lazy import):
```python
# monitor.py
class TrainingMonitor:
    def get_system_metrics(self) -> dict:
        metrics = {}
        try:
            import psutil
            metrics["cpu_utilization"] = psutil.cpu_percent(interval=None)
            metrics["cpu_memory"] = psutil.virtual_memory().percent
        except ImportError:
            pass
        # ... rest of method
```
Also do the same for the `tensorboard` import in `__init__`.

---

## HIGH (10)

### H1. CI proprietary leak check is incomplete

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/.github/workflows/ci.yml`, line 34
**What's wrong**: The CI grep pattern is:
```
dream11|databricks|dbfs|s3a://|mlflow|pacman|sporta|iceberg_catalog|accio|@dream11|dreamsports|nimbus
```
The plan's Step 12 master sweep additionally checks for: `s3://`, `dhruv.nigam@`, `susmit.saha@`, `palash.tatte@`, `infraautomation@`, `ghp_`, `AKCp8k`, `pypi-server.darwin`, `dream11-e2.cloud`, `ds_churn_temp`, `giveaway`, `supply_embed`, `supply_history`, `CrossAttentionForecaster`, `petastorm`, `pyspark`, `delta`.
**What it should be**: Match the full Step 12 pattern.
**Why it matters**: Future contributors could accidentally re-introduce proprietary references that CI would miss.

**Fix**: Replace the CI grep with the full pattern from the plan's Step 12.

---

### H2. Step 13 not completed — no topics or homepage URL set on GitHub

**Location**: GitHub repo settings for `dhruvnigam93/lumos-user-model`
**What's wrong**: Topics and homepage URL are both empty.
**What it should be**:
```bash
gh repo edit dhruvnigam93/lumos-user-model \
  --add-topic "transformer,user-modeling,multi-task-learning,pytorch,deep-learning,user-embeddings,behavior-prediction"
gh repo edit dhruvnigam93/lumos-user-model \
  --homepage "https://arxiv.org/abs/2512.08957"
```
**Why it matters**: Discoverability. Without topics, the repo won't appear in GitHub topic searches.

---

### H3. No Jupyter notebooks created

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/examples/` — only `.py` files
**What's wrong**: The plan (Step 9) says "Create quickstart.py, train_synthetic.py, extract_embeddings.py, and **Colab-ready notebooks**." The three `.py` scripts exist but zero `.ipynb` notebooks were created.
**What it should be**: At least one Colab-ready notebook (e.g., `examples/lumos_quickstart.ipynb`) with an "Open in Colab" badge in the README.
**Why it matters**: Colab notebooks dramatically lower the barrier to trying a library. Many ML researchers expect this. Also the plan explicitly specified it.

**Fix**: Create a notebook that mirrors `quickstart.py` + `train_synthetic.py`, with markdown cells explaining each step. Add a Colab badge at the top:
```python
# First cell:
!pip install git+https://github.com/dhruvnigam93/lumos-user-model.git
```

---

### H4. `LUMOSConfig` and `TrainingConfig` are dead code

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/lumos/config.py` (entire file)
**What's wrong**: These dataclasses are defined but never consumed anywhere:
- `LUMOS.__init__` takes 22 individual kwargs — no `from_config(config: LUMOSConfig)` class method exists
- `LUMOSTrainer.__init__` also takes raw kwargs — no config-based constructor
- No test, example, or internal code references these classes
**What it should be**: Either:
- (a) Add `LUMOS.from_config(config: LUMOSConfig)` and `LUMOSTrainer.from_config(...)` class methods that consume these dataclasses, OR
- (b) Remove the config module entirely and document that users pass kwargs directly
**Why it matters**: Dead code confuses users who discover it and expect to use it.

**Fix (recommended — option a)**: Add to `LUMOS`:
```python
@classmethod
def from_config(cls, config: "LUMOSConfig") -> "LUMOS":
    return cls(
        dim_activity_raw=config.dim_activity_raw,
        dim_static_raw=config.dim_static_raw,
        # ... all fields
    )
```

---

### H5. DDP/multi-GPU training silently dropped without documentation

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/lumos/training/trainer.py`
**What's wrong**: The original `training_ddp.py` had full DistributedDataParallel support (multi-GPU, all-reduce, DDP model wrapping). The extracted `LUMOSTrainer` is single-GPU only — hardcoded to a single device on line 49. This limitation is not mentioned anywhere.
**What it should be**: At minimum, document in the README and trainer docstring: "Note: This trainer supports single-GPU training only. For multi-GPU DDP training, wrap the model and data loaders manually."
**Why it matters**: Users reading the paper (which likely describes large-scale training) will expect multi-GPU support.

---

### H6. Non-standard gradient update ordering

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/lumos/training/trainer.py`, lines 151–155
**What's wrong**: The ordering is:
```python
loss.backward()
optimizer.step()       # step BEFORE zero_grad
scheduler.step()
optimizer.zero_grad()  # zero_grad AFTER step
```
The standard PyTorch pattern is `zero_grad()` → `backward()` → `step()`.
**What it should be**: Move `optimizer.zero_grad()` before `loss.backward()`:
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
if scheduler: scheduler.step()
```
**Why it matters**: While functionally equivalent on the first iteration (gradients start at zero), it's fragile. Any future modification that computes gradients between `step()` and `zero_grad()` would cause silent gradient accumulation bugs. It also confuses contributors familiar with the standard pattern.

---

### H7. Zero test coverage for `LUMOSTrainer`

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/tests/` — no `test_training.py`
**What's wrong**: The entire training loop (`train()`, `_validate()`, `_save_checkpoint()`, `_compute_loss_and_metrics()`) is untested. The plan (Step 8) only specified `test_model.py`, `test_losses.py`, `test_data.py`, `test_embeddings.py` — but this was an oversight.
**What it should be**: At minimum a smoke test that:
1. Creates a small model and synthetic dataset
2. Runs 2 epochs
3. Asserts loss is finite and decreasing
4. Asserts checkpoint file was saved
**Why it matters**: Training loop bugs (like H6's gradient ordering) go undetected without trainer tests.

---

### H8. Zero test coverage for evaluation metrics

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/tests/` — no `test_metrics.py`
**What's wrong**: `evaluate_binary()`, `evaluate_continuous()`, and `compute_ece()` contain substantial numeric logic (ECE binning, TPR@k percentile computation, within-k-percent, F1 threshold search) with no tests.
**What it should be**: Tests with known inputs/outputs:
- `compute_ece` with perfectly calibrated predictions should return ~0
- `evaluate_binary` with perfect predictions should return AUC=1.0
- `evaluate_continuous` with zero error should return RMSE=0, R2=1.0
**Why it matters**: Numeric errors in evaluation metrics are silent — they produce wrong numbers, not crashes.

---

### H9. Unused imports in evaluation/metrics.py

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/lumos/evaluation/metrics.py`, lines 6–7
**What's wrong**: `roc_curve` and `auc` are imported from `sklearn.metrics` but never used.
**What it should be**: Remove the unused imports.
**Why it matters**: Sloppy code signals poor review quality. Ruff would catch this if CI ran it (see L6).

**Fix**: Change line 3 to:
```python
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    r2_score,
)
```

---

### H10. Attention masking paths completely untested

**Location**: `/Users/dhruvnigam/Projects/lumos-oss/lumos/model/attention.py`, lines 29–30
**What's wrong**: `MultiHeadAttention.forward` accepts a `mask` parameter, and the masking codepath (`scores.masked_fill(mask == 0, float("-inf"))`) is propagated through all layers. No test ever passes a mask.
**What it should be**: At least one test that passes a mask and verifies it changes the output (vs. no mask).
**Why it matters**: Masking is critical for variable-length sequences, which is a real-world usage pattern. An untested mask path could silently break.

---

## MEDIUM (19)

### M1. `TrainingMonitor.get_system_metrics()` is dead code

**Location**: `lumos/training/monitor.py`, lines 22–38
**What**: Defined but never called from `LUMOSTrainer`.
**Fix**: Either call it from the training loop (as the original `training_ddp.py` did) or remove it.

### M2. `best_primary_metric` is dead code

**Location**: `lumos/training/trainer.py`, line 73
**What**: `self.best_primary_metric = 0.0` is initialized but never read or updated.
**Fix**: Either implement best-model tracking (save `best_model.pt` when validation metric improves) or remove the attribute.

### M3. Validation returns only loss, no per-task metrics

**Location**: `lumos/training/trainer.py`, `_validate()` method (lines 184–202)
**What**: Calls `_compute_loss_and_metrics()` but discards the metrics with `_, _ = ...`. The original training loop logged per-task validation AUC/MAPE.
**Fix**: Return and log validation metrics alongside validation loss.

### M4. No best-model checkpointing

**Location**: `lumos/training/trainer.py`, `_save_checkpoint()` (lines 204–217)
**What**: Saves a checkpoint every epoch but doesn't track the best validation loss/metric.
**Fix**: Track `self.best_val_loss`, save `best_model.pt` when it improves.

### M5. `preprocessing.py` utilities not integrated or exported

**Location**: `lumos/data/preprocessing.py` and `lumos/data/__init__.py`
**What**: `MinMaxScaler`, `LogMinMaxScaler`, and `build_sequences` exist and are tested, but not used by `UserSequenceDataset`, the trainer, or any example. Also not exported from `data/__init__.py`.
**Fix**: Export from `__init__.py` and add a usage example in the docstring or an example script.

### M6. Some evaluation metrics from original are missing

**Location**: `lumos/evaluation/metrics.py`
**What**: Missing from original PredictionEvaluator: coverage tracking, `positives@k` metrics, `actual_median`/`pred_median`. These were pure Python computations, not Spark-dependent.
**Fix**: Port the missing pure-Python metrics.

### M7. `DecoderLayer.forward` accepts unused `self_mask` parameter

**Location**: `lumos/model/layers.py`, line 48
**What**: `def forward(self, x, enc_out, self_mask=None, cross_mask=None)` — `self_mask` is accepted but never used (self-attention was commented out in the original code).
**Fix**: Remove `self_mask` parameter, or add a comment explaining it's reserved for future causal self-attention.

### M8. `MinMaxScaler.transform()` doesn't check `self.fitted`

**Location**: `lumos/data/preprocessing.py`, `transform()` method
**What**: Calling `transform()` before `fit()` produces a confusing `TypeError` instead of a clear error.
**Fix**: Add `if not self.fitted: raise RuntimeError("Call fit() before transform()")`.

### M9. `LogMinMaxScaler` silently destroys sign information

**Location**: `lumos/data/preprocessing.py`, lines 49, 53
**What**: Uses `np.abs(arr)` before log-transform, so negative values become positive. No documentation warns about this.
**Fix**: Add a docstring note: "Warning: This scaler applies abs() before log-transform. Sign information is lost."

### M10. `scikit-learn` is a core dependency but only used in training/evaluation

**Location**: `pyproject.toml`, line 37
**What**: `scikit-learn>=1.3` is required for all installs. Inference-only users (who only want `LUMOS` and embedding extraction) must install it unnecessarily.
**Fix**: Move to `[training]` optional deps. Guard the import in `trainer.py` and `evaluation/metrics.py`.

### M11. Bare `except Exception` in evaluation metrics

**Location**: `lumos/evaluation/metrics.py`, line 94
**What**: Silently catches all exceptions in TPR@k computation.
**Fix**: Catch specific exceptions (`ValueError`, `IndexError`).

### M12. `-1.0` sentinel value is ambiguous

**Location**: `lumos/training/trainer.py`, lines 86, 98–99
**What**: Uses `-1.0` as sentinel for "metric computation failed", but this could theoretically be a valid MAPE value.
**Fix**: Use `float('nan')` instead.

### M13. `NoEmbedding.__init__` type annotation conflict

**Location**: `lumos/model/positional.py`, line 88
**What**: `def __init__(self, d_model: int = None, max_len: int = None)` — `None` default conflicts with `int` type hint.
**Fix**: Change to `d_model: int | None = None, max_len: int | None = None`.

### M14. Duplicated column definitions across `synthetic.py` and `dataset.py`

**Location**: `lumos/data/synthetic.py` (lines 5–41) and `lumos/data/dataset.py` (lines 7–43)
**What**: Activity, static, event context, and target column lists are independently defined in both files with different variable names.
**Fix**: Create a shared `lumos/data/constants.py` with canonical column lists; import from both files.

### M15. `training/schedulers.py` has zero test coverage

**Location**: `lumos/training/schedulers.py`
**What**: `create_lr_scheduler()` supports 4 types. None tested.
**Fix**: Test that each scheduler type instantiates and that `step()` changes the LR.

### M16. Embedding reduction tests only assert shape, not semantics

**Location**: `tests/test_model.py`, lines 144–152
**What**: Tests for "mean", "max", "expavg", "last" only check output shape is `(2, 64)`. Doesn't verify reductions produce different results.
**Fix**: Run all 4 reductions on the same input and assert they produce different outputs.

### M17. Top-level `lumos/__init__.py` exports nothing useful

**Location**: `lumos/__init__.py`
**What**: Only `__version__`. Users can't do `from lumos import LUMOS`.
**Fix**: Add convenience imports: `from lumos.model import LUMOS` and `from lumos.config import LUMOSConfig`.

### M18. Missing return type annotations on public trainer methods

**Location**: `lumos/training/trainer.py`
**What**: `_build_loss_aggregator`, `_compute_loss_and_metrics`, `train` lack return type annotations.
**Fix**: Add `-> dict`, `-> tuple[torch.Tensor, list[float]]`, etc.

### M19. No column validation at `UserSequenceDataset` construction time

**Location**: `lumos/data/dataset.py`, `__init__`
**What**: Invalid column names only surface as `KeyError` deep in `__getitem__`, not at construction time.
**Fix**: Validate all specified columns exist in the DataFrame during `__init__`.

---

## LOW (15)

### L1. Lambda assigned to variable (PEP 8 E731)
**Location**: `lumos/model/lumos.py`, line 60 — `hidden = lambda d, depth: ...`
**Fix**: Replace with `def hidden(d, depth): return [d] * (depth - 1) if depth > 1 else []`

### L2. `gc.collect()` every 100 steps with hardcoded interval
**Location**: `lumos/training/trainer.py`, lines 170–171
**Fix**: Remove or make configurable.

### L3. Citation year may be wrong
**Location**: `README.md`, BibTeX block — `year={2024}`
**What**: arXiv ID `2512.08957` prefix `25` suggests December 2025, not 2024.
**Fix**: Verify against actual paper submission date and correct.

### L4. `loguru` in optional training deps but never imported
**Location**: `pyproject.toml`, line 45
**Fix**: Remove `loguru>=0.7` from dependencies.

### L5. CI tests only Python 3.10/3.11, local dev uses 3.13
**Location**: `.github/workflows/ci.yml`, matrix
**Fix**: Add `"3.12"` to the matrix. Consider `"3.13"` too.

### L6. CI does not run `ruff check` despite ruff being configured
**Location**: `.github/workflows/ci.yml` and `pyproject.toml` (ruff config)
**Fix**: Add `ruff check lumos/ tests/ examples/` step to CI.

### L7. `self.step` in `TrainingMonitor` is set but never read
**Location**: `lumos/training/monitor.py`, line 14
**Fix**: Remove.

### L8. Magic numbers without named constants
**Location**: `1e-8` in `extractor.py`, `1e-10` in `metrics.py`, `0.02` in `positional.py`
**Fix**: Define as module-level constants (e.g., `_EPS = 1e-8`).

### L9. `count_parameters` not exported from `evaluation/__init__.py`
**Location**: `lumos/evaluation/__init__.py`
**Fix**: Add to `__all__` and imports.

### L10. Preprocessing utilities not exported from `data/__init__.py`
**Location**: `lumos/data/__init__.py`
**Fix**: Add `MinMaxScaler`, `LogMinMaxScaler`, `build_sequences` to exports.

### L11. `TrainingMonitor` not exported from `training/__init__.py`
**Location**: `lumos/training/__init__.py`
**Fix**: Add to exports.

### L12. `UserSequenceDataset` stores redundant DataFrame references
**Location**: `lumos/data/dataset.py`, lines 88–95
**What**: `self.samples` stores `(uid, group_df, start_idx)` — the same group DataFrame reference for every window. Could store just user-level indices into `self.df` instead.
**Fix**: Refactor to store `(start_row_in_df, end_row_in_df)` tuples.

### L13. Missing `forward()` docstrings in model submodules
**Location**: `attention.py`, `embeddings.py`, `layers.py`, `positional.py`
**Fix**: Add input/output tensor shape documentation to each `forward()`.

### L14. `pyarrow` as core dependency but only used for optional parquet I/O
**Location**: `pyproject.toml`, line 38
**Fix**: Move to optional deps or document why it's required.

### L15. `TrainingMonitor.log_metrics` silently skips non-numeric values
**Location**: `lumos/training/monitor.py`, lines 16–19
**Fix**: Log a warning when a non-numeric value is passed.

---

## Proprietary Leak Audit: CLEAN

A comprehensive grep across all source files (`.py`, `.toml`, `.md`, `.yml`) found **zero proprietary references**. Specifically verified clean:

- No `dream11`, `d11`, `dreamsports`, `sporta`, `pacman`, `nimbus`
- No `databricks`, `dbfs`, `delta`, `iceberg_catalog`, `accio`
- No `mlflow`
- No `pyspark`, `spark`, `petastorm`
- No S3 paths (`s3://`, `s3a://`)
- No internal emails (`@dream11.com`, personal emails)
- No credentials (`ghp_`, `AKCp8k`, API keys)
- No internal URLs (`pypi-server.darwin`, `dream11-e2.cloud`)
- No internal code names (`supply_embed`, `supply_history`, `CrossAttentionForecaster`, `ds_churn_temp`)
- Synthetic data feature names are generic (session_count, purchase_amount, etc.) — no domain leakage
- Git history clean: single auto-generated LICENSE commit

---

## Architectural Fidelity: VERIFIED

Side-by-side comparison confirms the core model is a faithful port:

| Aspect | Original (`model_cross_attn.py`) | Extracted (`lumos/model/lumos.py`) | Match? |
|--------|----------------------------------|-------------------------------------|--------|
| MultiHeadAttention | Separate Q/K/V dims, scaled dot-product | Identical | Yes |
| EncoderLayer | Post-norm (residual → dropout → norm) | Identical | Yes |
| DecoderLayer | Cross-attention only (self-attention commented out) | Identical | Yes |
| ConfigurableEmbedding | Variable-depth MLP | Identical | Yes |
| Positional Embeddings | 5 types (none, sinusoidal, learned, tape, abs) | Identical | Yes |
| Forward Pass | embed → concat → project → encode → decode → output | Identical | Yes |
| `historical_user_embedding` | 4 reductions (mean, max, expavg, last) | Identical | Yes |
| `supply_embedding` | Direct embedding call | Renamed to `event_context_embedding` | Yes |

**Naming changes applied correctly**:
- `CrossAttentionForecaster` → `LUMOS`
- `supply_embed` → `event_context_embed`
- `user_embed` → `activity_embed`
- `dim_supply_raw` → `dim_event_context_raw`
- `supply_history` → `event_context_history`
- `future_supply` → `future_event_context`

**Note**: The plan mentioned extracting from `dev-naman-2` branch for "SwiGLU + pre-norm" but neither the original code (checked `model_cross_attn.py` on the current branch) nor the extraction has SwiGLU or pre-norm — both use ReLU + post-norm. This plan item was either an aspiration that was never merged, or refers to a branch not checked out.

---

## Recommended Fix Order

For a developer picking this up fresh, here is the suggested order:

### Phase 1: Blockers (must do before any push)
1. **C2** — Fix LICENSE copyright (2 minutes)
2. **C4** — Update `.gitignore`, clean build artifacts (5 minutes)
3. **C5** — Guard `psutil`/`tensorboard` imports with try/except (15 minutes)
4. **C3** — Verify paper param count, fix `from_paper_config()` if needed (30 minutes)
5. **H9** — Remove unused imports in `metrics.py` (2 minutes)
6. **H6** — Fix gradient ordering in trainer (5 minutes)
7. **L4** — Remove unused `loguru` dependency (2 minutes)
8. **C1** — Commit and push everything (5 minutes)

### Phase 2: Quality (do before sharing publicly)
9. **H1** — Expand CI proprietary grep (10 minutes)
10. **H2** — Set GitHub topics and homepage (5 minutes)
11. **H4** — Wire up config dataclasses or remove them (30 minutes)
12. **H5** — Document single-GPU limitation in README (10 minutes)
13. **H7** — Write trainer smoke test (30 minutes)
14. **H8** — Write evaluation metrics tests (30 minutes)
15. **H10** — Write masking test (15 minutes)
16. **L3** — Verify and fix citation year (5 minutes)
17. **L5/L6** — Add Python 3.12 to CI matrix, add ruff step (10 minutes)

### Phase 3: Polish (do when convenient)
18. **M14** — Deduplicate column definitions (20 minutes)
19. **M17** — Add convenience imports to top-level `__init__.py` (5 minutes)
20. **M7/M13** — Fix unused parameters and type annotations (10 minutes)
21. **H3** — Create Colab notebook (45 minutes)
22. **M2/M3/M4** — Implement best-model tracking and validation metrics (1 hour)
23. Remaining MEDIUM and LOW items as time permits

---

## Appendix: Quick-Reference Commands

### Run tests
```bash
cd /Users/dhruvnigam/Projects/lumos-oss
uv run python -m pytest tests/ -v
```

### Run proprietary leak check locally
```bash
grep -ri "dream11\|databricks\|dbfs\|s3a://\|s3://\|mlflow\|pacman\|sporta\|iceberg_catalog\|accio\|@dream11\.com\|d11-data\|dreamsports\|nimbus\|dhruv\.nigam@\|susmit\.saha@\|palash\.tatte@\|ghp_\|AKCp8k\|dream11-e2\.cloud\|ds_churn_temp\|giveaway\|supply_embed\|supply_history\|CrossAttentionForecaster\|petastorm\|pyspark" lumos/ tests/ examples/ || echo "CLEAN"
```

### Verify model instantiation
```bash
uv run python -c "from lumos.model import LUMOS; m = LUMOS.from_paper_config(); print(f'Params: {sum(p.numel() for p in m.parameters()):,}')"
```

### Run full training example
```bash
uv run python examples/train_synthetic.py
```

### Set GitHub topics and homepage
```bash
gh repo edit dhruvnigam93/lumos-user-model \
  --add-topic "transformer,user-modeling,multi-task-learning,pytorch,deep-learning,user-embeddings,behavior-prediction" \
  --homepage "https://arxiv.org/abs/2512.08957"
```
