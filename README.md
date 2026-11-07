# TCAV as an Auditing Signal

Code and experimental evidence for **Auditing Learned Visual Shortcuts with
TCAV**.

This repository studies whether TCAV concept sensitivity can serve as evidence
of learned shortcut reliance. It contains three stages: a tiger/striped
interpretability study, a concept/control/layer stability study, and a controlled
CIFAR-10 shortcut audit. The archived audit used one model per condition.
The extension trained 15 models across five seeds and evaluated two shortcut settings.

## Layout

| Path | Purpose |
| --- | --- |
| `experiments/preliminary/` | Code extracted from the executed ResNet-18 notebook |
| `experiments/stability/` | Stability code, helpers and the archived 22-row table |
| `experiments/controlled/` | Data preparation, training, behavioral tests, TCAV and archived results |
| `tools/run_stage.py` | Stage launcher with isolated controlled runs |
| `tools/paper_results.py` | Dependency-free evaluation of the manuscript decision rule |
| `tests/` | Decision-boundary and raw-output consistency checks |
| `experiments/controlled/run_replicated.py` | Five-seed, two-setting experiment |
| `experiments/controlled/results/replicated/` | Per-model metrics, protocol, and rate summaries |

Historical learning folders and private research notes remain local and are
excluded by `.gitignore`. Model weights, image datasets and CAV caches are not
included in the public tree. Existing Git-tracked files require separate index
cleanup; ignore rules alone do not remove tracked files.

## Verify the archived result

From the repository root, using a configured Python interpreter:

```bash
python tools/paper_results.py
python -m unittest discover -s tests -v
```

These commands require only the Python standard library. They do not download
data, train networks or rerun TCAV. The evaluator uses the manuscript rule:
**layer3 patch sign count >= 0.7 for at least two of three controls**.

| Model | Layer3 scores | Decision relative to training-condition reference |
| --- | --- | --- |
| Clean | 0.155, 1.000, 1.000 | False positive |
| Spurious | 0.995, 0.925, 0.520 | True positive |

The confusion matrix is TP=1, FP=1, TN=0, FN=0. A high sign count measures
concept-direction sensitivity; it does not establish causal reliance. The clean
model is a negative reference because it was trained without a patch/class
association, not because it is invariant to patch insertion.

## Run the experiments

The research scripts require the imports in `requirements.txt`. This file adds
previously omitted direct dependencies; it is not a historical version lock.
The archived notebooks record Python 3.10, but exact historical library versions
are unavailable. The checkpoint recheck used Python 3.12.14 and the CPU
environment documented in the controlled experiment guide.

```bash
python -m pip install -r requirements.txt
python tools/run_stage.py download --dry-run
```

Controlled stages require a separate run directory, for example
`python tools/run_stage.py testsets --run-dir out/recheck`. Use a new directory
for each protocol or model pair. See the [controlled experiment guide](experiments/controlled/README.md)
for the stage sequence, checkpoint inputs, and the distinction between archived
results and the recheck.

The controlled experiment guide also gives the command for reproducing the
five-seed extension and describes how to interpret its model-level intervals.
