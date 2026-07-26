# Set2Proto-Diffusion

Minimal research infrastructure for testing discrete, evidence-guided prototype
generation from sets of face features. The repository is being built in explicit
stages so that every training run starts from a validated, reproducible
environment.

## Current status

Stages 1 through 5 are implemented:

- one MVP configuration with `smoke`, `pilot`, and `full` profiles;
- validated configuration/profile merging;
- deterministic Python, NumPy, PyTorch, CUDA, and DataLoader seeding;
- collision-safe run and checkpoint paths;
- JSONL events, CSV metrics, and an atomic run manifest;
- CUDA/BF16/native-SDPA environment validation.
- identity-disjoint synthetic `[49, 512]` FP16 feature banks;
- disjoint low-quality condition sets and high-quality teacher sets;
- deterministic feature-space corruption with visibility metadata;
- clean, low-quality, complementary-occlusion, common-occlusion, and
  wrong-identity scenarios;
- an explicit self-teacher baseline and CUDA BF16 DataLoader smoke probe.
- train-only 512→128 PCA projection stored as portable tensors;
- quality/local-consensus robust teacher prototypes;
- K=1024 spherical MiniBatch K-means and per-split discrete target maps;
- reconstruction cosine, codebook utilization, and token perplexity reports.
- a parameter-matched one-shot/MaskGIT Conditional Transformer;
- four-step confidence, evidence-ordering, evidence-logit, and remask decoding;
- BF16 training, exact RNG/optimizer checkpoints, and ordered OOM fallback;
- 1/2/4/8-step sampling and frame-permutation smoke evaluation.
- projected-condition precomputation for repeated training epochs;
- best-frame, mean, max, and quality-weighted continuous baselines;
- identity-verification ROC-AUC, EER, and TAR at configured FAR targets;
- a 2,000-step parameter-matched pilot profile.

No dataset, face-recognition checkpoint, or pretrained weight is downloaded by
this stage. The backbone remains out of scope until a local licensed dataset and
AdaFace weight path, or explicit download permission, are supplied.

## Requirements

The checked local environment is:

- Windows, Python 3.10;
- PyTorch with CUDA support;
- PyYAML and NumPy;
- an NVIDIA GPU for the default configuration.

The project currently relies only on packages already present in the workspace.
It does not require `pytest`; tests use Python's standard `unittest`.

## Environment check

Run from the repository root in PowerShell:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile smoke --stage env-check
```

The command creates a new directory under `outputs/`, runs a small BF16 native
scaled-dot-product-attention operation, and prints a compact JSON result. It does
not start a training job.

Every run contains:

```text
outputs/<timestamp>_<profile>_s<seed>/
├── artifacts/
├── checkpoints/
└── logs/
    ├── events.jsonl
    ├── metrics.csv
    └── run_manifest.json
```

An existing run is never overwritten implicitly. To revisit an existing run,
provide its exact id together with `--resume`. A future checkpoint at optimizer
step 125 is named `checkpoint_step_00000125.pt`.

Useful optional overrides:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile smoke --stage env-check --seed 1234
python scripts/run_mvp.py --config configs/mvp.yaml --profile full --stage env-check --run-id local-full-check
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover configuration rejection, profile merging, repeatable RNG/DataLoader
ordering, collision-safe run creation, checkpoint naming, structured logs,
identity split isolation, S/T separation, corruption semantics, and exact
synthetic-data reproducibility.

## Synthetic feature data

Generate and validate one profile without running a GPU batch:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile smoke --stage prepare-synthetic
```

Generate the smoke profile and transfer a real `[B, 4, 49, 512]` condition and
teacher batch to CUDA in BF16:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile smoke --stage synthetic-smoke
```

Each command creates a collision-safe dataset under:

```text
data/synthetic/<run-id>/
├── manifest.json
├── train_features.pt
├── train_sets.jsonl
├── val_features.pt
├── val_sets.jsonl
├── test_features.pt
└── test_sets.jsonl
```

The tensor banks contain clean unit-normalized local features, globally unique
identity ids, per-image quality values, and AdaFace-norm surrogates. JSONL rows
contain deterministic condition/teacher indices, scenario seeds, and explicit
distractor indices. The default disjoint teacher uses the highest-quality images
of an identity; `SyntheticSetDataset(..., teacher_mode="self")` exposes the
self-teacher baseline.

These are deliberately feature-space surrogates for pipeline validation:
blur smooths the 7×7 grid, low light mixes feature noise, JPEG quantizes feature
values, and occlusions replace hidden tokens while recording exact visibility.
They are not substitutes for image-level real-data evidence.

## Projection, teacher targets, and codebook

Fit PCA and the codebook from the training split only, then generate targets for
all three identity-disjoint splits:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile smoke --stage fit-codebook --dataset-root data/synthetic/<dataset-run-id>
```

Artifacts are written without overwrite under:

```text
cache/quantization/<run-id>/
├── manifest.json
├── projection.pt
├── codebook.pt
├── train_targets.pt
├── val_targets.pt
└── test_targets.pt
```

Teacher pooling computes local frame consensus at every spatial position,
combines it with the quality-norm surrogate, trims the lowest-scoring frame, and
normalizes the weighted result. The saved target files contain both continuous
128-dimensional prototypes and their K-means token maps.

The initial smoke generator assigns independent high-dimensional directions to
unseen identities. Consequently, a codebook fitted on only 24 synthetic train
identities is expected to reconstruct held-out identities poorly. This is
reported as a failed research gate, not hidden and not treated as evidence that
real AdaFace features will fail. The synthetic run validates the mechanics; the
quantization hypothesis requires licensed real features.

## Parameter-matched training smoke

Run the four-layer, hidden-256 one-shot and MaskGIT models for the two optimizer
steps defined by the smoke profile:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile smoke --stage train-smoke `
  --dataset-root data/synthetic/<dataset-run-id> `
  --quantization-root cache/quantization/<quantization-run-id>
```

Both models instantiate the same `ConditionalTokenTransformer` and start from
the same state dictionary. The only training difference is the corruption:
one-shot always receives 49 mask tokens, while MaskGIT samples a cosine mask
ratio and computes loss only at masked positions.

The Transformer uses target self-attention and cross-attention over all
condition-frame local tokens. It has spatial embeddings but deliberately has no
frame-index embedding. Deterministic runs force PyTorch's native math-SDPA
backend because memory-efficient CUDA attention backward is nondeterministic on
the current Windows build.

Checkpoints include model, optimizer, AMP scaler, batch generator, CPU RNG, and
CUDA RNG state. If training encounters CUDA OOM, a new recorded attempt follows
the configured order: batch size, condition-frame count, then hidden dimension.
One-shot and MaskGIT always use the same selected fallback configuration.

The smoke profile is a functional test, not a convergence experiment. Its two
steps are expected to have near-random K=1024 accuracy and must not be used to
claim one method is better.

## Synthetic convergence pilot

The pilot uses its own 80/20/20 identity-disjoint data and train-only codebook.
After preparing those artifacts, run:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot --stage train-pilot `
  --dataset-root data/synthetic/<pilot-data-run-id> `
  --quantization-root cache/quantization/<pilot-quantization-run-id>
```

The pilot performs 2,000 optimizer steps for each parameter-matched model. Every
step uses four gradient-accumulation micro-batches. Conditions are projected
once into memory before training, avoiding repeated 512→128 PCA work.

Verification uses the clean disjoint-teacher prototype of every held-out
identity as a common gallery. Each method's clean and corrupted set prototypes
are queries against that gallery. This produces scenario-specific ROC-AUC, EER,
and TAR values with the exact positive/negative pair counts and an explicit FAR
reliability flag.

The first synthetic pilot showed real optimization convergence, but it did not
support multi-step superiority: one-shot and one-step decoding outperformed
four-step confidence MaskGIT. Evidence-logit guidance improved confidence-only
decoding under several corruptions, while remasking was not consistently
beneficial. Mean and quality pooling were perfect on this simple synthetic
identity construction. These are diagnostic results, not claims about AdaFace
features.

## Configuration profiles

- `smoke`: two future optimizer steps, batch size 2, and a compact synthetic
  identity split. It retains the target four-layer/256-hidden model and K=1024
  codebook so structural mistakes are visible.
- `pilot`: a 2,000-step intermediate run with batch size 4.
- `full`: the target batch size 8, accumulation 4, and 10k–20k step schedule.

All profiles inherit from `defaults` in the single
[`configs/mvp.yaml`](configs/mvp.yaml). Resolved paths are based on the
repository location, not the caller's current directory, which keeps Windows
invocations stable.

## CelebA integrity audit and identity split

After the user has accepted the CelebA non-commercial research agreement and
placed the official files under `data/real/celeba`, run:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile full --stage prepare-real --dataset-root data/real/celeba --run-id stage6-real-audit
```

The stage verifies the official annotation MD5 values, confirms that all
202,599 annotations cover 10,177 identities, rejects identity leakage, and
writes deterministic `identities.csv` and `images.csv` manifests. The full
profile selects 1,000 train, 100 validation, and 250 test identities with at
least ten images per identity. The pilot profile uses 200/25/50 identities.

The stage is intentionally reported as `blocked` until either the official
`img_align_celeba.zip` passes MD5 or all 202,599 aligned images are present.
Google Drive folder downloads may produce misleading partial archives; the raw
`img_celeba.7z` distribution is complete only when volumes `.001` through
`.014` are present.

## Frozen AdaFace backbone smoke

Download the official AdaFace R50 WebFace4M checkpoint from the
[author-provided Google Drive link](https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view)
and save it as:

```text
weights/adaface/adaface_ir50_webface4m.ckpt
```

Then run the real-image backbone smoke before any feature cache:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile smoke --stage backbone-smoke --dataset-root data/real/celeba --run-id stage7-backbone-smoke
```

The loader strictly checks checkpoint keys, freezes all parameters, applies a
deterministic five-landmark similarity transform to 112x112 BGR input, and
automatically discovers the last spatial feature map before the embedding
flattening step. The observed shape, feature norm, peak CUDA memory, checkpoint
SHA-256, command, configuration, and environment are written to the run log.

## Offline real-feature cache

After the backbone smoke passes, cache frozen spatial and global AdaFace
features. The smoke profile intentionally limits each split to 16 images:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile smoke `
  --stage cache-real-features --dataset-root data/real/celeba `
  --run-id stage8-feature-cache-smoke
```

Each split stores normalized `[N, 49, 512]` local features, normalized
`[N, 512]` embeddings, raw AdaFace quality norms, and an index CSV under
`cache/real_features/<run-id>/`. Arrays use NumPy `.npy` memmaps and float16
feature storage. Progress is committed after every batch. An interrupted run
can continue without replacing completed rows:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile smoke `
  --stage cache-real-features --dataset-root data/real/celeba `
  --run-id stage8-feature-cache-smoke --resume
```

Completion validates shapes, dtypes, finite values, unit normalization, index
row counts, SHA-256 hashes, the discovered spatial hook, and peak GPU memory.

## Real pilot, diagnostics, and report

Create image-space perturbed condition sets and disjoint clean teachers from
the pilot cache:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage prepare-real-sets --dataset-root data/real/celeba `
  --feature-cache-root cache/real_features/stage8-feature-cache-pilot `
  --run-id stage9-real-sets-pilot
```

Fit the train-only PCA-128 projection and K=1024 spherical codebook, run the
mandatory two-step real smoke, then run the parameter-matched pilot:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage fit-codebook --dataset-root data/real_sets/stage9-real-sets-pilot `
  --run-id stage10-real-codebook-pilot

python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage train-real-smoke `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --run-id stage11-real-training-smoke

python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage train-pilot `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --run-id stage13-real-training-pilot-2k
```

Complete latency, commit/visibility, failure-case, plot, and Go/No-Go outputs:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage evaluate-diagnostics `
  --training-run outputs/stage13-real-training-pilot-2k `
  --run-id stage14-real-diagnostics

python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage build-report `
  --training-run outputs/stage13-real-training-pilot-2k `
  --diagnostics-run outputs/stage14-real-diagnostics `
  --run-id stage15-final-report
```

The current pilot decision is recorded in
`outputs/stage15-final-report/artifacts/go_no_go.json`. It is a No-Go for
scaling the current discrete representation to 10k-20k steps: evidence logits
helped every tested scenario, but quality pooling remained much stronger,
remasking was inconsistent, and local K=1024 reconstruction failed its gate.
The recommended next experiment is deterministic evidence-guided local
aggregation.

## P0-1 deterministic condition robust pooling

Evaluate the teacher's unchanged local-consensus, AdaFace-quality, trim, and
softmax rule on the condition set. This stage reads the stage13 manifest only
to regression-check the existing gallery protocol; it does not load a model
or checkpoint:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage evaluate-condition-pooling `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --training-run outputs/stage13-real-training-pilot-2k `
  --run-id p0-condition-robust-pooling-reviewed
```

The collision-safe run writes `REPORT.md`, core/scenario/weight CSV tables,
`diagnostics.json`, and a reproducibility manifest under
`outputs/p0-condition-robust-pooling-reviewed/`. Validation and test both use the
existing clean disjoint-teacher continuous gallery. PCA and codebook files are
checksum-checked before and after evaluation and are never refit.
Because run directories are collision-safe, use a fresh `--run-id` for a
second execution.

## P0-2 visibility/reliability-aware deterministic pooling

Build a train-only positionwise clean-teacher reference bank, select the
predeclared reliability/identity-gate grid on validation, persist and reload a
selection lock, and then evaluate exactly one locked configuration on test:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage evaluate-visibility-aggregation `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --training-run outputs/stage13-real-training-pilot-2k `
  --run-id p0-2-visibility-reliability-pooling-reviewed
```

The reviewed run is under
`outputs/p0-2-visibility-reliability-pooling-reviewed/`. Its train-only bank
contains 800 clean references with layout `[49,800,128]`; validation locked
`top_k=16` and identity weight `1.0`. The mechanism downweighted wrong-ID
frames and preferred visible complementary regions, but it did not beat
quality pooling on validation hard AUC. The metadata oracle also lacked
validation headroom, while test moved in the opposite direction. This is a
No-Go for further tuning at the current late AdaFace hook. The next bounded
experiment is the same metadata-oracle check on an earlier spatial hook; no
Transformer or diffusion retraining is justified unless that check shows
validation and test headroom. See `artifacts/REPORT.md`,
`validation_search.csv`, and `diagnostics.json` in the reviewed run.

Dataset and checkpoint provenance remain recorded in run manifests. The
project never downloads or accepts third-party dataset licenses on the user's
behalf.

## P0-3 earlier-hook metadata-oracle headroom

Evaluate whether perfect condition visibility/source-identity metadata has
enough headroom when raw activations are pooled before a frozen AdaFace suffix.
The four hooks and all validation gates are fixed in `configs/mvp.yaml`; the
existing PCA, codebook, clean disjoint-teacher gallery, split, and checkpoint
are only read and checksum-checked:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage evaluate-earlier-hook-oracle `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --run-id p0-3-earlier-hook-oracle-rerun
```

Use a fresh run ID because output directories are collision-safe. The completed
reviewed run is `outputs/p0-3-earlier-hook-oracle-reviewed-v4/`. On validation,
none of `body.3`, `body.6`, `body.7`, or `body.20` passed the preregistered
headroom gates. The strongest candidate, `body.20`, reached hard macro AUC
0.957683 versus 0.974083 for late quality pooling; it was lower in all four
hard scenarios. The lock therefore records `selected=null` and
`test_authorized=false`, and the run does not construct the test token dataset,
load test images, or create a test early-feature cache.

This is a No-Go for training an earlier-hook visibility estimator on the
current AdaFace representation. The reviewed run passed all provenance,
suffix/cache replay, unit-norm, permutation, protected-output, and 7.2 GiB
memory checks; peak CUDA reserved memory was 604 MiB. Earlier `reviewed`,
`reviewed-v2`, and `reviewed-v3` directories are retained failed harness-debug
runs and are not the final scientific artifact.

## P1-0 continuous residual oracle headroom

Measure whether the existing late continuous condition features contain
recoverable information beyond quality pooling before training another shared
model. The teacher-guided methods are intentionally non-deployable: at each
position they may retain the quality anchor, choose the teacher-closest observed
frame, optimize a convex combination, or add a norm-bounded residual in the
frame-disagreement span.

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage evaluate-continuous-residual-oracle `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --run-id p1-0-continuous-residual-oracle-rerun
```

The completed run is
`outputs/p1-0-continuous-residual-oracle-reviewed-v2/`. Validation selected the
teacher-guided best-frame-or-quality-anchor oracle: hard macro AUC increased
from 0.974083 to 0.990083. The persisted lock then authorized exactly that
method on test, where hard AUC increased from 0.997316 to 0.999849. All four
hard-scenario deltas were positive; complementary occlusion had the largest
test gain at +0.008171. The oracle is not a usable inference method because it
requires the disjoint teacher prototype for every probe. It establishes
headroom for a lightweight continuous routing/residual predictor, not evidence
for residual diffusion by itself.

The earlier output without the `-v2` suffix is retained as a failed
correctness-harness run and is not the final scientific artifact. Any learned
follow-up must use a new identity holdout because current test identities were
already observed in earlier stages.

## P1-1 trainable continuous local router

P1-1 uses the frozen cached and existing-PCA-projected condition features with
shape `[B,4,49,128]`. The teacher-guided best-frame-or-quality oracle creates
training supervision only. Inference uses condition features, AdaFace quality
norms, and within-set local/global consensus; it does not read the teacher,
execute the backbone, or refit PCA/codebook artifacts.

Run the mandatory two-step smoke first. Smoke never constructs the test
dataset:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage train-continuous-router-smoke `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --run-id p1-1-continuous-router-smoke
```

After smoke and unit tests pass, run the formal pilot:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage train-continuous-router `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --run-id p1-1-continuous-router-pilot
```

Formal training runs for at most 2,000 steps and evaluates validation every
100 steps. Checkpoints are ranked only by validation hard AUC, pooled-all AUC,
then teacher-map cosine. The test dataset is constructed only if all
preregistered validation gates pass, and exactly one locked checkpoint is
evaluated. Results, training history, routing-weight diagnostics, and
reproducibility records are written under the new run's `artifacts/` and
`logs/` directories.

## P1-2 identity-gated anchor residual router

P1-2 initializes from the validation-selected P1-1 checkpoint, preserves
feature-norm quality pooling as a deterministic anchor, and learns a
per-position residual gate. Training adds direct identity cross-entropy and a
hardest-impostor cosine-margin loss against a train-only clean
disjoint-teacher gallery. Neither teacher nor gallery is used at inference.
Two preregistered loss recipes are trained independently and selected only on
validation.

Run the two-step, two-recipe smoke:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage train-identity-gated-router-smoke `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --initial-router-checkpoint outputs/p1-1-continuous-router-pilot-reviewed/checkpoints/checkpoint_step_00000300.pt `
  --run-id p1-2-identity-gated-router-smoke
```

Then run the formal validation-gated pilot:

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage train-identity-gated-router `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --initial-router-checkpoint outputs/p1-1-continuous-router-pilot-reviewed/checkpoints/checkpoint_step_00000300.pt `
  --run-id p1-2-identity-gated-router-pilot
```

The test dataset remains unconstructed unless one recipe passes every
preregistered validation gate. The selected checkpoint, candidate comparison,
gate diagnostics, split metrics, and provenance hashes are stored in the new
run directory without modifying P1-1 or earlier artifacts.

## P1-3 bounded scalar-evidence router

P1-3 addresses P1-2 gate overfitting directly. The gate no longer receives
raw anchor/routed identity vectors; it receives 13 identity-agnostic scalars
covering route entropy, top-score gap, quality concentration, route-versus-
quality divergence, local/global frame consensus, and anchor disagreement.
The residual gate is capped at `0.35`, and the gallery objective asks the
output to improve its identity margin relative to the deterministic quality
anchor. Two candidates compare a frozen P1-2 router with low-rate router
fine-tuning.

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage train-scalar-evidence-router-smoke `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --initial-router-checkpoint outputs/p1-2-identity-gated-router-pilot-reviewed/checkpoints/identity_balanced/checkpoint_step_00000100.pt `
  --run-id p1-3-scalar-evidence-router-smoke
```

```powershell
python scripts/run_mvp.py --config configs/mvp.yaml --profile pilot `
  --stage train-scalar-evidence-router `
  --dataset-root data/real_sets/stage9-real-sets-pilot `
  --quantization-root cache/quantization/stage10-real-codebook-pilot `
  --initial-router-checkpoint outputs/p1-2-identity-gated-router-pilot-reviewed/checkpoints/identity_balanced/checkpoint_step_00000100.pt `
  --run-id p1-3-scalar-evidence-router-pilot
```

As in P1-1/P1-2, test remains unavailable until a validation-selected
candidate passes every fixed scientific and correctness gate.

## P2-1 anchor-relative residual tokens and two-level evidence

P2-1 retains the existing PCA and disjoint-teacher targets byte-for-byte. It
uses continuous quality pooling as the condition anchor, fits a train-only raw
Euclidean K=1024 codebook to `teacher - anchor`, and reconstructs every token
as `normalize(anchor + residual_code)`. Residual centroids are deliberately
not unit-normalized and must not be loaded with the legacy spherical-codebook
loader.

Two-level evidence first estimates a permutation-equivariant global
identity-inlier weight per frame, then combines it with candidate-specific
local cosine support. For residual tokens, support is computed against the
sample-dependent final candidate `normalize(anchor + residual_code)`, not the
residual direction alone.

The complete command performs train-only codebook fitting, representation
gating, mandatory two-step smoke, parameter-matched 2,000-step one-shot and
MaskGIT training, validation evaluation, latency, memory, and permutation
checks:

```powershell
python scripts/run_p2_residual_evidence.py `
  --config configs/mvp.yaml `
  --profile expanded `
  --stage all `
  --dataset-root data/real_sets/stage16-expanded-real-sets `
  --absolute-quantization-root cache/quantization/stage16-expanded-quantization `
  --residual-artifact-root cache/residual_quantization/p2-1-expanded-residual `
  --run-id p2-1-residual-evidence-validation
```

This stage is validation-only by construction: it does not instantiate the
Stage16 test dataset and refuses to create `test_targets.pt`. The actual report
is in
`outputs/p2-1-residual-evidence-validation/artifacts/REPORT.md`. A future
paper-level decision must lock the P2 design first and then build a new
identity-disjoint holdout.
