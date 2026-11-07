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

def transform(img: Image.Image) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return tf(img)

def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transform(img)


def load_image_tensors(class_name, root_path, apply_transform=True):
    path = os.path.join(root_path, class_name)
    files = glob.glob(os.path.join(path, "*.JPEG"))

    images = []
    for f in files:
        img = Image.open(f).convert("RGB")
        images.append(transform(img) if apply_transform else img)
    return images


def assemble_concept(name, cid, concepts_path):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    data_iter = dataset_to_dataloader(dataset)
    return Concept(id=cid, name=name, data_iter=data_iter)

concepts_path = "data/tcav_exp/image/concepts/"

banded = assemble_concept("banded", 0, concepts_path)
cracked = assemble_concept("cracked", 1, concepts_path)
dotted = assemble_concept("dotted", 2, concepts_path)
fibrous = assemble_concept("fibrous", 3, concepts_path)
lined = assemble_concept("lined", 4, concepts_path)

random_0 = assemble_concept("random_0", 5, concepts_path)
random_1 = assemble_concept("random_1", 6, concepts_path)
random_2 = assemble_concept("random_2", 7, concepts_path)
random_3 = assemble_concept("random_3", 8, concepts_path)

imagenet_root = "data/tcav_exp/image/imagenet/"
tiger_imgs = load_image_tensors("tiger", imagenet_root, apply_transform=False)
tiger_tensors = torch.stack([transform(img) for img in tiger_imgs])

model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

labels = ResNet18_Weights.DEFAULT.meta["categories"]
tiger_ind = labels.index("tiger")

layers = ["layer2", "layer3"]

tcav = TCAV(
    model=model,
    layers=layers,
    layer_attr_method=LayerIntegratedGradients(
        model, None, multiply_by_inputs=False
    )
)

experimental_sets_1 = [
    [banded, random_0],
    [banded, random_1],
    [lined, random_0],
    [lined, random_1],
]

start = time.time()
tcav_scores = tcav.interpret(
    inputs = tiger_tensors,
    experimental_sets = experimental_sets_1,
    target = tiger_ind,
    n_steps = 5
)
print("Time taken:", time.time() - start, "s")

def format_float(f):
    return float("{:.3f}".format(f)) if abs(f) >= 5e-4 else float("{:.3e}".format(f))


def plot_tcav_scores(experimental_sets, tcav_scores):
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

plot_tcav_scores(experimental_sets_1, tcav_scores)

experimental_set_2 = [
    [
        banded,
        cracked,
        dotted,
        fibrous,
        lined,
        random_0,
        random_1,
        random_2,
        random_3,
    ]
]

start = time.time()
tcav_scores_2 = tcav.interpret(
    inputs = tiger_tensors,
    experimental_sets = experimental_set_2,
    target = tiger_ind,
    n_steps = 5
)
end = time.time()
print("time taken: ", end - start)

plot_tcav_scores(experimental_set_2, tcav_scores_2)

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

experimental_sets_3 = [
    [banded, random_0],
    [banded, random_1],
    [banded, random_2],
    [banded, random_3],

    [random_0, random_1],
    [random_0, random_2],
    [random_0, random_3],
]

tcav_scores_3 = tcav.interpret(
    inputs = tiger_tensors,
    experimental_sets = experimental_sets_3,
    target = tiger_ind,
    n_steps = 5
)

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

for layer in layers:
    show_boxplots(
        scores = tcav_scores_3,
        experimental_sets = experimental_sets_3,
        layer = layer,
        metric = "sign_count",
        n_per_plot = 4
    )
