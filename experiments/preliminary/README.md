# Preliminary tiger/striped study

`run.py` contains the code cells of the executed historical
`TCAV_experiment.ipynb`, in their original order. The incomplete standalone
export and the GoogleNet/zebra tutorial are excluded from the public tree.

The model is ImageNet-pretrained ResNet-18. The target is tiger, and the legacy
concept name `banded` refers to the same striped image set described in the
manuscript. Inputs are resized to 256, center-cropped to 224, and ImageNet
normalized. TCAV uses Layer Integrated Gradients, `multiply_by_inputs=False`,
five steps, and layer2/layer3. The notebook also contains exploratory
multi-concept comparisons; not every plot is a manuscript result.

Run `python tools/run_stage.py preliminary` from the repository root after
providing the original images below this folder:

```text
data/tcav_exp/image/
  concepts/{banded,cracked,dotted,fibrous,lined,random_0,random_1,random_2,random_3}/
  imagenet/tiger/*.JPEG
```

These images are not present in this checkout. No substitute images were
generated. The historical plotting code uses floor division to group boxplots;
seven comparisons with groups of four leave the last three comparisons out of
the displayed groups. That behavior is preserved. Statistical results are
exploratory and do not constitute a causal test or a multiple-testing correction.
