## Reading Summary

### Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference (McCoy et al., 2019)
- The paper shows that many NLI models give correct answers by using simple syntactic patterns, rather than truly understanding sentence meaning.
- The authors describe three common heuristics: lexical overlap, subsequence, and constituent heuristics.
- To test these heuristics, the paper introduces the HANS dataset, which includes examples where such heuristics give wrong answers.
- Even though models perform well on MNLI, many of them perform very poorly on HANS, especially on non-entailment cases.
- When training data includes examples that break these heuristics, model performance improves, showing that training data strongly affects model behavior.

### Shortcut Learning in Deep Neural Networks (Geirhos et al., 2020)
- Many deep neural networks achieve high accuracy by using shortcuts, which are simple patterns in the data that do not reflect the real task.
- This helps explain why models can do well on standard test data but fail when the data distribution changes.
- Shortcut learning is influenced by several factors together, including model design, training data, and training objectives.
- Common benchmarks often cannot reveal shortcut learning, so additional robustness tests are needed.
- As a result, interpretability methods may point to shortcut features instead of the true concepts the model should learn.

### The Mythos of Model Interpretability (Lipton, 2018)
- The paper argues that the word “interpretability” is used in many different ways and does not have a single clear meaning.
- It distinguishes between transparency, which means understanding how a model works, and post-hoc explanations, which are added after the model makes a decision.
- Linear models are not always easy to understand, especially when they use many features or complex preprocessing.
- Post-hoc explanations may look reasonable, but they do not always reflect how the model actually makes decisions.
- Claims about interpretability should clearly state their goal, who the explanation is for, and how it should be evaluated.
