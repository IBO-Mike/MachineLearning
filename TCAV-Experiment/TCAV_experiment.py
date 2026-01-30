import numpy as np
import os, glob
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import ttest_ind
import time

# PyTorch 及 torchvision 相关库
import torch
import torchvision
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights

# Captum 中与可解释性和 TCAV 相关的模块
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str


# 将 PIL Image 转换为 ResNet18 可接受的标准化 Tensor
def transform(img: Image) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    return tf(img)


# 从图像文件路径读取图片，并转换为 Tensor
def get_tensor_from_filename(filename: str) -> torch.Tensor:
    img = Image.open(filename).convert("RGB")
    return transform(img)


# 加载某一类别的所有图片
# 默认返回 Tensor 列表，用作 TCAV 的输入样本
def load_image_tensors(class_name: str, root_path="data/tcav_exp/image/imagenet/", transform=True):
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(os.path.join(path, "*.JPEG"))
    tensors = []
    for filename in filenames:
        img = Image.open(filename)
        if transform:
            tensors.append(transform(img))
        else:
            tensors.append(img)
    return tensors


# 构造 TCAV 所需的 Concept 对象
# name: 概念名称（对应文件夹名）
# id: 概念的唯一编号
def assemble_concept(name, id, concept_path="data/tcav_exp/image/concepts/"):
    concept_path = os.path.join(concept_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)
    concept = Concept(id=id, name=name, data_iter=concept_iter)
    return concept


# 辅助函数：格式化浮点数，便于展示
def format_float(f):
    if abs(f) >= 0.0005:
        return float("{:.3f}".format(f))
    else:
        return float("{:.3e}".format(f))


# 绘制 TCAV 结果柱状图
# experimental_sets: 概念组合
# tcav_scores: TCAV 返回的结果字典
def plot_tcav_scores(experimental_sets, tcav_scores):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize=(25, 7))
    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):
        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)
        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i - 1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [
                format_float(scores["sign_count"][i])
                for layer, scores in tcav_scores[concepts_key].items()
            ]
            _ax.bar(pos[i], val, width=barWidth, edgecolor="white", label=concepts[i].name)
        _ax.set_xlabel("Set {}".format(str(idx_es)), fontweight="bold", fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)
        _ax.legend(fontsize=16)
    plt.show()


# 概念图片路径
concepts_path = "data/tcav_exp/image/concepts/"

# 构造多个概念（纹理类 + 随机对照）
banded_concept = assemble_concept("banded", 0, concepts_path)
cracked_concept = assemble_concept("cracked", 1, concepts_path)
dotted_concept = assemble_concept("dotted", 2, concepts_path)
fibrous_concept = assemble_concept("fibrous", 3, concepts_path)
lined_concept = assemble_concept("lined", 4, concepts_path)

random_0_concept = assemble_concept("random_0", 5, concepts_path)
random_1_concept = assemble_concept("random_1", 6, concepts_path)
random_2_concept = assemble_concept("random_2", 7, concepts_path)
random_3_concept = assemble_concept("random_3", 8, concepts_path)

# 加载目标类别 tiger 的图片
tiger_imgs = load_image_tensors("tiger", transform=False)
tiger_tensors = torch.stack([transform(img) for img in tiger_imgs])

# ImageNet 类别标签
labels = ResNet18_Weights.DEFAULT.meta["categories"]

# 检查 tiger 是否正确
print(labels[292])
print(labels.index("tiger"))
tiger_ind = labels.index("tiger")

# 可视化各概念对应的示例图片（仅用于展示）
n_figs = 5
n_concepts = 9

fig, axs = plt.subplots(n_concepts, n_figs + 1, figsize=(5 * n_figs, 4 * n_concepts))

for c, concept in enumerate(
    [
        banded_concept,
        cracked_concept,
        dotted_concept,
        fibrous_concept,
        lined_concept,
        random_0_concept,
        random_1_concept,
        random_2_concept,
        random_3_concept,
    ]
):
    concept_path = os.path.join(concepts_path, concept.name) + "/"
    img_files = glob.glob(os.path.join(concept_path, "*"))
    for i, img_file in enumerate(img_files[: n_figs + 1]):
        if os.path.isfile(img_file):
            if i == 0:
                axs[c, i].text(
                    1.0, 0.5, str(concept.name),
                    ha="right", va="center", family="sans-serif", size=24
                )
            else:
                img = plt.imread(img_file)
                axs[c, i].imshow(img)
            axs[c, i].axis("off")

# 加载 ResNet18 模型，并指定用于 TCAV 的中间层
model = torchvision.models.resnet18(pretrained=True)
model.eval()
layers = ["layer2", "layer3"]

# 构建 TCAV 实验对象
# 使用 LayerIntegratedGradients 作为层归因方法
tcav_exp = TCAV(
    model=model,
    layers=layers,
    layer_attr_method=LayerIntegratedGradients(
        model, None, multiply_by_inputs=False
    ),
)

# 实验 1：单一概念 + 随机对照
experimental_set_1 = [
    [banded_concept, random_0_concept],
    [banded_concept, random_1_concept],
    [lined_concept, random_0_concept],
    [lined_concept, random_1_concept],
]

start = time.time()
tcav_scores_1 = tcav_exp.interpret(
    inputs=tiger_tensors,
    experimental_sets=experimental_set_1,
    target=tiger_ind,
    n_steps=25,
)
end = time.time()

print("time taken: ", end - start)
plot_tcav_scores(experimental_set_1, tcav_scores_1)

# 实验 2：多个概念同时参与对比
experimental_set_2 = [
    [
        banded_concept,
        cracked_concept,
        dotted_concept,
        fibrous_concept,
        lined_concept,
        random_0_concept,
        random_1_concept,
        random_2_concept,
        random_3_concept,
    ]
]

start = time.time()
tcav_scores_2 = tcav_exp.interpret(
    inputs=tiger_tensors,
    experimental_sets=experimental_set_2,
    target=tiger_ind,
    n_steps=5,
)
end = time.time()
print("time taken: ", end - start)
plot_tcav_scores(experimental_set_2, tcav_scores_2)

# 绘制 boxplot，用于比较不同概念的 TCAV 分布
def show_boxplots(
    scores,
    experimental_sets,
    layer,
    metric="sign_count",
    n_per_plot=4,
    ylim=(0, 1),
):
    def label_names(exp_set):
        return [
            exp_set[0].name,
            exp_set[1].name.split("_")[0] + "_rand",
        ]

    n_plots = len(experimental_sets) // n_per_plot
    fig, axes = plt.subplots(1, n_plots, figsize=(25, 6))
    if n_plots == 1:
        axes = [axes]

    for i in range(n_plots):
        es_slice = experimental_sets[i * n_per_plot : (i + 1) * n_per_plot]

        P1, P2, pval, relation = get_pval(
            scores=scores,
            experimental_sets=es_slice,
            score_layer=layer,
            score_type=metric,
            alpha=0.05,
            print_ret=False,
        )

        axes[i].boxplot([P1, P2], showfliers=True)
        axes[i].set_ylim(ylim)
        axes[i].set_title(
            f"{layer} | {metric}\n(p = {pval}, {relation})",
            fontsize=16,
        )
        axes[i].set_xticklabels(label_names(es_slice[0]), fontsize=14)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)

    plt.show()

# 对每个层绘制 boxplot（示例）
for layer in layers:
    show_boxplots(
        scores=tcav_scores_3,
        experimental_sets=experimental_sets_3,
        layer=layer,
        metric="sign_count",
        n_per_plot=4,
    )