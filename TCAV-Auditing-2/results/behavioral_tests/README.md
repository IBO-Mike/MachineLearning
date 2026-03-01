## Evaluation Metrics

**Detection rule:**  
At layer3, if the patch TCAV score ≥ 0.7 in at least 2 out of 3 control sets,  
the model is considered to rely on the patch.

### Ground Truth
- Clean model: does NOT rely on patch  
- Spurious model: DOES rely on patch  

### Detection Outcome

| Model     | Detected as relying on patch | Ground Truth | Result |
|------------|-----------------------------|--------------|--------|
| Clean      | Yes                         | No           | False Positive (FP) |
| Spurious   | Yes                         | Yes          | True Positive (TP)  |

### Confusion Matrix

- **TP = 1**  
- **FN = 0**  
- **FP = 1**  
- **TN = 0**