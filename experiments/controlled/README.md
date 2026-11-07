# Controlled CIFAR-10 patch audit

`results/` contains the archived tables and raw scores. New controlled runs use
a separate directory. The original scripts remain in the local historical
`TCAV-Auditing-2/` tree. Use Python 3.10 or later with the dependencies listed
in the repository root.

## Recheck existing checkpoints

Run from the repository root:

```sh
python tools/run_stage.py download --run-dir out/recheck
python tools/run_stage.py prepare --run-dir out/recheck
python tools/run_stage.py concept --run-dir out/recheck
python tools/run_stage.py testsets --run-dir out/recheck
```

Place the checkpoints at `out/recheck/models/{clean_model,spurious_model}/model.pth`.
If training from scratch, run `train-clean` and `train-spurious` with the same
`--run-dir`. A run can supply its own `configs/train_clean.yaml` and
`configs/train_spurious.yaml` to record a different seed.

```sh
python tools/run_stage.py behavior --run-dir out/recheck
python tools/run_stage.py tcav --run-dir out/recheck
python tools/run_stage.py summarize --run-dir out/recheck
python tools/paper_results.py --table out/recheck/results/tables/tcav_scores_table.csv
```

The recheck generates deterministic controls with seed 42 and sorts image paths
before sampling concept and audit images. The Python, NumPy, and PyTorch seeds for concept fitting are fixed per model. TCAV uses a fresh cache per model
invocation and saves input and checkpoint hashes in the raw JSON. The flat
export contains the patch entry from Captum's `sign_count` and `magnitude`
arrays. The paired behavioral summary reports cat-prediction gains, losses,
and mean cat-logit changes against the original images. The experimental unit
for model-level audit outcomes is the checkpoint.

## Protocol

- CIFAR-10 cat (class 3), ResNet-18 initialized without pretrained weights.
- Red 8x8 bottom-right patch on the shortcut-trained model's target-class
  training images. Training uses 30 epochs, SGD with learning rate 0.1,
  momentum 0.9, weight decay 0.0005, batch size 128, seed 42, random crop and
  horizontal flip. The checkpoint with the lowest training loss is retained.
- Behavioral evaluation uses the same test images under original, target
  patched, and target adjacent-block conditions. The adjacent-block
  intervention copies pixels from the block to the left of the patch.
- The concept uses 200 patched training images sampled across classes, with
  three separately sampled 200-image unpatched controls. TCAV evaluates up
  to 200 unpatched test cats, four layers, and CPU tensor inputs.
- The manuscript rule uses layer3, an inclusive threshold of 0.7, and at
  least two passing controls. The old `evaluation_metrics.py` uses a
  different layer-mean rule and is retained as a historical script.

## Observed checkpoint recheck

The run at `out/icmlc2027-recheck/` reused the two archived model checkpoints,
using Python 3.12.14, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, Captum 0.9.0,
and scikit-learn 1.9.1. CIFAR-10 was restored from the Python distribution
matching torchvision's archive MD5. The local environment record is
`out/tcav-runtime.txt`.

| Run | Clean layer3 | Shortcut-trained layer3 | Outcomes |
| --- | --- | --- | --- |
| Archive | 0.155, 1.000, 1.000 | 0.995, 0.925, 0.520 | FP, TP |
| Recheck, CAV seed 42 | 0.955, 0.995, 0.925 | 1.000, 1.000, 1.000 | FP, TP |
| Same inputs, CAV seed 43 | 0.115, 1.000, 0.955 | 1.000, 1.000, 1.000 | FP, TP |

Reversing model execution order at seed 42 reproduced all 24 model-layer-control
scores. The seed 43 diagnostic reused the identical samples and checkpoints and
wrote its outputs to the run folder under `results/cav_seed_43/`.

The six overall and cat accuracy pairs reproduce. Red-patch insertion on
1,000 matched cat images yields 56 gains and 249 losses for the clean
checkpoint, and 998 gains with zero losses for the shortcut-trained checkpoint.
The mean cat-logit changes are −0.452 and +13.828, respectively.

The original random-control sample IDs and software versions are unavailable.
The archived script used shared default Captum cache identifiers. The recheck
changes the samples, cache behavior, and environment together. Its results
cannot identify which change accounts for score differences across runs. It
uses the same two trained checkpoints, so it provides no additional
independent training runs.

## Independent training extension

Run the four-block CNN experiment from the repository root in the same CPU
environment. Pass the directory containing the verified CIFAR-10 Python batches.

~~~sh
python -m experiments.controlled.run_replicated --data out/icmlc2027-recheck/data/raw --output out/icmlc2027-replicated-v2
~~~

Training seeds 101–105 each produce a clean checkpoint and two shortcut
checkpoints: red patch on cats and checkerboard on airplanes. The two settings
share each seed's clean checkpoint. The script evaluates each setting separately
with the historical TCAV voting rule and a paired input intervention baseline.
The baseline flags a model when patch insertion increases the fraction of
non-target test images predicted as the target by at least 10 percentage points.
The output includes checkpoint histories, sample IDs and hashes, paired effects,
per-model scores, and exact model-level binomial intervals. The run's protocol
record fingerprints the script and software environment.

The completed run trained 15 unique checkpoints and produced 20 audits.
Per-model metrics, the protocol, and the interval summaries are available in
[`results/replicated/`](results/replicated/).

| Setting | TCAV false positives | Intervention false positives | True positives (each detector) |
| --- | --- | --- | --- |
| Red patch / cat | 3/5 | 0/5 | 5/5 |
| Checkerboard / airplane | 0/5 | 1/5 | 5/5 |

The exact 95% confidence interval is 14.7–94.7% for 3/5, 0–52.2% for 0/5,
0.5–71.6% for 1/5, and 47.8–100% for 5/5. These intervals concern repeated
training within each fixed setting. The clean models are shared across settings.

The initial architecture pilot placed a linear prediction head directly after
the audited layer. The completed protocol adds a nonlinear fourth block and
trains new checkpoints. Pilot runs are stored separately from the reported run.

