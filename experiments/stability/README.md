# Concept, control and layer stability

`scripts/run.py` contains the code cells of the historical stability notebook.
`tcav_utils.py` and `tcav_plot.py` are copied unchanged. Only the final CSV
destination was adapted to `../results/stability_scores.csv` so it matches the
new directory structure. Notebook prose and outputs were not promoted into
executable code. `results/stability_scores.csv` is the byte-identical archived
22-row result table, not newly generated data.

Run from the repository root with `python tools/run_stage.py stability` after
providing the original image folders under `experiments/stability/data/`:

```text
target/tiger/*.JPEG
concepts/striped_A/
concepts/striped_B/
control_pool/<control-name>/
```

The study uses ImageNet-pretrained ResNet-18, tiger inputs, ImageNet preprocessing,
Layer Integrated Gradients with five steps, and:

1. A versus B concepts at layer2/layer3. The implementation directly contrasts
   A against B (and reverses their order); it does not compare both with one
   fixed neutral control.
2. Striped A against three sampled controls at layer2/layer3.
3. Striped A against three newly sampled controls at layer1 through layer4.

Experiment 3 samples controls again, despite the old notebook prose saying it
reuses experiment 2 controls. Sampling is not seeded. Cache directories are
reset for each subexperiment. Original concept/control images are absent from
this checkout, so the archived table supports numerical inspection but not
exact sample-level reproduction. Fresh execution overwrites the copied table.
