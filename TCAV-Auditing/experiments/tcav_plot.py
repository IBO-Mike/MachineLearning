import os, glob, time
import numpy as np
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


def format_float(f):
    return float("{:.3f}".format(f)) if abs(f) >= 5e-4 else float("{:.3e}".format(f))


def plot_tcav_scores(experimental_sets, tcav_scores, layers):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize=(25, 7))
    bar_width = 1 / (len(experimental_sets[0]) + 1)

    for idx, concepts in enumerate(experimental_sets):
        key = concepts_to_str(concepts)
        positions = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            positions.append([x + bar_width for x in positions[i-1]])

        cur_ax = ax[idx] if len(experimental_sets) > 1 else ax
        for i, concept in enumerate(concepts):
            values = [
                format_float(scores["sign_count"][i])
                for scores in tcav_scores[key].values()
            ]
            cur_ax.bar(positions[i], values, width=bar_width, label=concept.name)

        cur_ax.set_xticks([r + bar_width for r in range(len(layers))])
        cur_ax.set_xticklabels(layers)
        cur_ax.legend()

    plt.show()

def assemble_scores(scores, experimental_sets, idx, score_layer, score_type):
    score_list = []
    for concepts in experimental_sets:
        key = "-".join([str(c.id) for c in concepts])
        score_list.append(scores[key][score_layer][score_type][idx])
    return score_list


def get_pval(scores, experimental_sets, score_layer, score_type,
             alpha=0.05, print_ret=True):
    P1 = assemble_scores(scores, experimental_sets, 0, score_layer, score_type)
    P2 = assemble_scores(scores, experimental_sets, 1, score_layer, score_type)

    _, pval = ttest_ind(P1, P2)

    relation = "Disjoint" if pval < alpha else "Overlap"

    if print_ret:
        print(f"Layer: {score_layer}")
        print(f"P1 mean/std: {np.mean(P1):.3f} / {np.std(P1):.3f}")
        print(f"P2 mean/std: {np.mean(P2):.3f} / {np.std(P2):.3f}")
        print(f"p-value: {pval:.4f} → {relation}")
        print("-" * 40)

    return P1, P2, pval, relation

def show_boxplots(
    scores,
    experimental_sets,
    layer,
    metric="sign_count",
    n_per_plot=4,
    ylim=(0, 1)
):

    def label_names(exp_set):
        return [
            exp_set[0].name,
            exp_set[1].name.split("_")[0] + "_rand"
        ]

    n_plots = len(experimental_sets) // n_per_plot
    fig, axes = plt.subplots(1, n_plots, figsize = (25, 6))
    if n_plots == 1:
        axes = [axes]

    for i in range(n_plots):
        es_slice = experimental_sets[i * n_per_plot : (i + 1) * n_per_plot]

        P1, P2, pval, relation = get_pval(
            scores = scores,
            experimental_sets = es_slice,
            score_layer = layer,
            score_type = metric,
            alpha = 0.05,
            print_ret = False
        )

        axes[i].boxplot([P1, P2], showfliers = True)
        axes[i].set_ylim(ylim)
        axes[i].set_title(
            f"{layer} | {metric}\n(p = {pval}, {relation})",
            fontsize = 16
        )
        axes[i].set_xticklabels(label_names(es_slice[0]), fontsize = 14)
        axes[i].grid(axis = "y", linestyle = "--", alpha = 0.5)

    plt.show()