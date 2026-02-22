
---

# 📁 4️⃣ task1_llava_report.md

```markdown
# Task 1 – LLaVA Vision-Language Model Report

## Model Description

We evaluated LLaVA-1.5-7B, a large vision-language model,
for zero-shot pneumonia detection.

Unlike CNN-based approaches,
this model generates text responses to medical images.

## Methodology

- Dataset: PneumoniaMNIST
- Image resized to 224×224
- Prompt-based binary classification
- No fine-tuning performed
- Evaluated on 100 test samples

## Results

Accuracy: (Insert your result)

Classification report:
(Insert generated output)

## Analysis

Strengths:
- No task-specific training required
- General multimodal reasoning capability

Limitations:
- Not medically specialized
- May hallucinate explanations
- Computationally expensive
- Not optimized for diagnostic accuracy

## Comparison to CNN

CNN (ResNet18):
- Trained on dataset
- AUC ≈ 0.96
- Accuracy ≈ 0.89

LLaVA (Zero-shot):
- No training
- Expected lower accuracy
- Demonstrates multimodal reasoning

## Conclusion

Vision-language models show promise,
but domain-specific CNNs remain superior
for medical classification tasks.
