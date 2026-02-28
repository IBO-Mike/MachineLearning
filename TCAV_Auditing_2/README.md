# TCAV Spurious Patch Experiment

## Goal
Build a controlled setup with a known spurious feature (corner patch), verify true model reliance behaviorally, and audit it using TCAV. Evaluate auditing correctness with TP/TN/FP/FN.

## Fixed Setup
- Dataset: CIFAR-10
- Backbone: ResNet-18
- Target class: cat
- Spurious feature: red 8x8 patch at bottom-right corner
- Injection rule: add patch to *training images of target class only*

## Repository Layout
- data/raw: original dataset
- data/clean: clean training split
- data/spurious: patched training split
- data/concept_patch: concept set for TCAV (images containing patch)
- data/control_sets: multiple random control sets
- models: checkpoints
- results: behavioral test + TCAV scores + final tables

## Reproducibility
- Use fixed random seeds.
- All outputs go under `results/`.