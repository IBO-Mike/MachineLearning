import os, glob, time
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import ttest_ind

import torch
import torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from captum.attr import LayerIntegratedGradients
from captum.concept import TCAV, Concept
from captum.concept._utils.data_iterator import (
    dataset_to_dataloader, CustomIterableDataset
)
from captum.concept._utils.common import concepts_to_str

from tcav_utils import *
from tcav_plot import *



model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

TARGET_IMAGE_ROOT = "../data/target/"
tiger_imgs = load_image_tensors("tiger", TARGET_IMAGE_ROOT, apply_transform=False)
tiger_tensors = torch.stack([transform(img) for img in tiger_imgs])

labels = ResNet18_Weights.DEFAULT.meta["categories"]
tiger_ind = labels.index("tiger")

concepts_path = "../data/concepts/"

striped_A = assemble_concept("striped_A", 0, concepts_path=concepts_path)
striped_B = assemble_concept("striped_B", 1, concepts_path=concepts_path)

control_pool_path = "../data/control_pool/"


experiment1_layers = ["layer2", "layer3"]

CAV_PATH_EXP1 = "./cav_exp1"
reset_cav_dir(CAV_PATH_EXP1)

tcav_exp1 = TCAV(
    model=model,
    layers=experiment1_layers,
    layer_attr_method=LayerIntegratedGradients(
        model, None, multiply_by_inputs=False
    ),
    save_path=CAV_PATH_EXP1
)

experimental_sets_1 = [
    [striped_A, striped_B],
    [striped_B, striped_A]
]

tcav_scores_exp1 = tcav_exp1.interpret(
    inputs=tiger_tensors,
    experimental_sets=experimental_sets_1,
    target=tiger_ind,
    n_steps=5
)

plot_tcav_scores(experimental_sets_1, tcav_scores_exp1, experiment1_layers)

exp1_records = extract_sign_scores(
    tcav_scores_exp1,
    experimental_sets_1,
    experiment1_layers
)

df_exp1 = pd.DataFrame(exp1_records)
df_exp1["experiment"] = "Concept Stability"
df_exp1["control_set_id"] = "N/A"

experiment2_layers = ["layer2", "layer3"]

CAV_PATH_EXP2 = "./cav_exp2"
reset_cav_dir(CAV_PATH_EXP2)

tcav_exp2 = TCAV(
    model=model,
    layers=experiment2_layers,
    layer_attr_method=LayerIntegratedGradients(
        model, None, multiply_by_inputs=False
    ),
    save_path=CAV_PATH_EXP2
)

controls_exp2 = sample_random_controls(
    pool_path=control_pool_path,
    n_controls=3
)

experimental_sets_2 = [
    [striped_A, rc] for rc in controls_exp2
]

tcav_scores_exp2 = tcav_exp2.interpret(
    inputs=tiger_tensors,
    experimental_sets=experimental_sets_2,
    target=tiger_ind,
    n_steps=5
)

plot_tcav_scores(experimental_sets_2, tcav_scores_exp2, experiment2_layers)

exp2_records = extract_sign_scores(
    tcav_scores_exp2,
    experimental_sets_2,
    experiment2_layers
)

df_exp2 = pd.DataFrame(exp2_records)
df_exp2["experiment"] = "Control Sensitivity"

df_exp2["concept_set"] = "striped_A"

df_exp2["control_set_id"] = (
    df_exp2.groupby(["layer"])
    .cumcount()
)

summary_exp2 = (
    df_exp2
    .groupby("layer")["sign_score"]
    .agg(["mean", "std"])
    .reset_index()
)

summary_exp2

plt.figure(figsize=(6, 4))
df_exp2.boxplot(
    column="sign_score",
    by="layer",
    grid=False
)
plt.title("Experiment 2")
plt.suptitle("")
plt.ylabel("TCAV Score")
plt.show()

experiment3_layers = ["layer1", "layer2", "layer3", "layer4"]

CAV_PATH_EXP3 = "./cav_exp3"
reset_cav_dir(CAV_PATH_EXP3)

tcav_exp3 = TCAV(
    model=model,
    layers=experiment3_layers,
    layer_attr_method=LayerIntegratedGradients(
        model, None, multiply_by_inputs=False
    ),
    save_path=CAV_PATH_EXP3
)

controls_exp3 = sample_random_controls(
    pool_path=control_pool_path,
    n_controls=3
)

experimental_sets_3 = [
    [striped_A, rc] for rc in controls_exp3
]

tcav_scores_exp3 = tcav_exp3.interpret(
    inputs=tiger_tensors,
    experimental_sets=experimental_sets_3,
    target=tiger_ind,
    n_steps=5
)

plot_tcav_scores(
    experimental_sets_3,
    tcav_scores_exp3,
    experiment3_layers
)

exp3_records = extract_sign_scores(
    tcav_scores_exp3,
    experimental_sets_3,
    experiment3_layers
)

df_exp3 = pd.DataFrame(exp3_records)
df_exp3["experiment"] = "Layer Sensitivity"
df_exp3["concept_set"] = "striped_A"
df_exp3["control_set_id"] = (
    df_exp3.groupby(["layer"])
    .cumcount()
)

summary_exp3 = (
    df_exp3
    .groupby("layer")["sign_score"]
    .agg(["mean", "std"])
    .reset_index()
)

summary_exp3

plt.figure(figsize=(7, 4))
plt.errorbar(
    summary_exp3["layer"],
    summary_exp3["mean"],
    yerr=summary_exp3["std"],
    capsize=4,
    marker="o"
)
plt.title("Experiment 3: Layer Sensitivity (Mean ± Std across Controls)")
plt.xlabel("Layer")
plt.ylabel("TCAV Sign Score")
plt.show()

audit_table = pd.concat(
    [df_exp1, df_exp2, df_exp3],
    ignore_index=True
)

final_audit_table = audit_table[
    [
        "experiment",
        "layer",
        "concept_set",
        "control_set_id",
        "sign_score"
    ]
].rename(columns={
    "sign_score": "tcav_score"
})

final_audit_table.to_csv(
    "../results/stability_scores.csv",
    index=False
)

final_audit_table
