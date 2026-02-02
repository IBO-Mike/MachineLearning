## Reflection

### Stability of the TCAV Signal

Across all three experiments, the TCAV signal for the striped concept was
remarkably stable under controlled perturbations. When using different random
control sets, the TCAV sign scores for the concept of interest remained consistently
high, with only minor variations across runs. This stability was further supported
by the low variance observed in both boxplots and error-bar visualizations.

At the same time, TCAV was not completely invariant: shallow layers exhibited
slightly higher variance, indicating that early representations are more sensitive
to control choice and noise. Overall, however, the TCAV signal did not fluctuate
in a way that would undermine interpretability.

### Factors Affecting TCAV Results

Among the three factors examined—concept set, control set, and layer depth—the
network layer had the strongest and most systematic impact on TCAV scores.
Layer-wise analysis revealed a clear and monotonic increase in TCAV sign scores
from shallow to deeper layers.

In contrast, changing the control set had a comparatively smaller effect.
While absolute scores varied slightly across different random controls, the
relative ordering and overall magnitude of TCAV responses remained stable.
Differences between concept sets with the same semantics were minimal at
intermediate layers and only became noticeable at deeper layers.

### Implications for TCAV as an Auditing Signal

These results suggest that TCAV can serve as a useful auditing signal, but only
when its robustness is explicitly evaluated. The method appears reliable at
intermediate and deeper layers, where concept representations are more structured
and stable. However, TCAV results at shallow layers should be interpreted with
caution due to higher sensitivity to experimental choices.

Overall, this study highlights that TCAV should not be treated as a single
definitive explanation. Instead, its value lies in comparative and robustness-based
analysis, where stability across concepts, controls, and layers is used as evidence
for trustworthy interpretations.