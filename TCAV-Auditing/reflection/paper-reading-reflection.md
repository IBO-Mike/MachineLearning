## Reading Summary

### Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference (McCoy et al., 2019)
- The paper shows that many NLI models achieve correct predictions by relying on shallow syntactic heuristics rather than genuine semantic understanding.
- Three common heuristics are identified: lexical overlap, subsequence, and constituent heuristics, each allowing models to exploit surface patterns in sentence pairs.
- The authors introduce the HANS dataset, which is explicitly designed to break these heuristics by constructing cases where heuristic-based reasoning leads to incorrect conclusions.
- Despite strong performance on MNLI, many models fail dramatically on HANS, particularly on non-entailment examples, revealing a lack of robustness across evaluation settings.
- When training data is augmented with counterexamples that invalidate these heuristics, model behavior improves, highlighting the strong influence of data design on model reasoning strategies.

### Shortcut Learning in Deep Neural Networks (Geirhos et al., 2020)
- The paper argues that high accuracy in deep neural networks is often achieved through shortcut learning, where models exploit simple, task-irrelevant correlations in the data.
- Shortcut learning explains why models can perform well on standard benchmarks yet fail under distribution shifts or stress tests.
- The emergence of shortcuts is driven jointly by model architecture, training data, and optimization objectives, rather than by a single factor.
- Standard benchmarks are often insufficient to expose shortcut reliance, motivating the need for robustness-oriented evaluations.
- As a consequence, interpretability methods may consistently highlight shortcut features, meaning that stable explanations do not necessarily correspond to semantically meaningful or desired model behavior.

### The Mythos of Model Interpretability (Lipton, 2018)
- The paper argues that “interpretability” lacks a single, precise definition and is used to describe multiple, often conflated goals.
- It distinguishes between transparency, which concerns understanding the internal mechanics of a model, and post-hoc explanations, which aim to rationalize model outputs after the fact.
- Even seemingly simple models, such as linear models, may not be genuinely interpretable when they involve high-dimensional features or complex preprocessing.
- Post-hoc explanations can appear intuitive and convincing while failing to faithfully represent the true decision process of the model.
- Claims of interpretability should therefore clearly specify their purpose, target audience, and evaluation criteria, emphasizing the need to critically assess explanation methods themselves.
