# 🤖 Trustworthy AI Text Generator

> **Evaluating the trustworthiness of a simple text generation model (t5-small) for academic title generation.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PjKUoEHjvgagw-dQccsAvtrQA0q-Cjg5?usp=sharing)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

---

## 📋 Overview

This project evaluates whether a small generative AI model (t5-small) can produce useful academic paper titles while remaining **reproducible**, **grounded** in source abstracts, and **robust** to paraphrased inputs.

### 🎯 Main Research Question

> **Can a small generative AI model produce useful text while remaining reproducible, grounded, and robust?**

### 🔬 Trustworthiness Properties Evaluated

| Property | Question | Metric |
|----------|----------|--------|
| 🔁 **Reproducibility** | Does the model produce consistent outputs? | Exact-match rate across 10 runs |
| 📍 **Grounding** | Are generated titles supported by input? | Hallucination rate |
| 🛡️ **Robustness** | Does the model handle paraphrased inputs? | Word overlap similarity |

---

## ✨ Key Features

- **Model:** t5-small (~60M parameters) fine-tuned for abstract-to-title generation
- **5 Decoding Methods:** Greedy, Temperature (0.7, 1.0, 1.3), Top-k (k=10)
- **Dataset:** 100 real abstract-title pairs from CS/ML papers
- **Evaluation:** 15 held-out examples never used for training

---

## 📊 Dataset

| Split | Size | Purpose |
|-------|------|---------|
| Training | 70 | Model fine-tuning |
| Validation | 15 | Hyperparameter tuning |
| **Trust-Evaluation** | **15** | **NEVER used for training** |

**Data Statistics:**
- Average abstract length: ~120 words
- Average title length: 7.3 words (range: 5–11)
- All titles ≤ 32 tokens ✅ (compatible with t5-small)

---

## 🚀 Getting Started

### Option 1: Run on Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PjKUoEHjvgagw-dQccsAvtrQA0q-Cjg5?usp=sharing)

Click the badge above to open the notebook in Colab and run all experiments.

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/ManojKumar178/ai-text-generator.git
cd ai-text-generator

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook code/trust_ai_assignment.ipynb
```

### Option 3: Quick Start Code

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def generate_title(abstract):
    input_text = "summarize: " + abstract
    inputs = tokenizer(input_text, return_tensors='pt', 
                      truncation=True, max_length=256)
    outputs = model.generate(inputs.input_ids, 
                           max_length=64, min_length=2, 
                           do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Try it!
abstract = "This study proposes joint models for intent detection and slot filling in spoken language understanding."
title = generate_title(abstract)
print(title)
```

---

## 📁 Repository Structure

```
ai-text-generator/
├── code/
│   └── trust_ai_assignment.ipynb   # Main Colab notebook
├── dataset/
│   ├── train.csv                   # 70 training examples
│   ├── val.csv                     # 15 validation examples
│   └── eval.csv                    # 15 trust-evaluation examples
├── output/
│   ├── task1_generated_titles.csv  # Basic generation
│   ├── task2_generated_titles.csv  # All decoding methods
│   ├── task3_reproducibility.csv   # Reproducibility results
│   ├── task4_hallucination.csv     # Hallucination analysis
│   ├── task5_robustness.csv        # Robustness results
│   └── loss_curves.png             # Training/validation loss curves
├── README.md
└── requirements.txt
```

---

## 📈 Key Results

### 🔁 Task 3: Reproducibility

| Decoding Method | Exact-Match Rate | Near-Duplicate Rate | Stability |
|-----------------|------------------|---------------------|-----------|
| Greedy | **100%** | **100%** | **High** |
| Temperature 0.7 | 13% | 33% | Low |
| Temperature 1.0 | 10% | 15% | Low |
| Temperature 1.3 | 10% | 10% | Low |
| Top-k (k=10) | 10% | 14% | Low |

> **💡 Insight:** Greedy decoding achieves **100% reproducibility** because it's deterministic — no randomness in token selection.

---

### 📍 Task 4: Hallucination

| Decoding Method | Grounded Rate | Partially Grounded | Hallucination Rate |
|-----------------|---------------|-------------------|-------------------|
| Greedy | **73%** | 27% | **0%** |
| Temperature 0.7 | 53% | 33% | 13% |
| Temperature 1.0 | 0% | 13% | **87%** |
| Temperature 1.3 | 7% | 13% | 80% |
| Top-k (k=10) | 20% | 33% | 47% |

> **💡 Insight:** Hallucination increases with temperature. **Greedy = 0% hallucination.** Temperature 1.0 produces the most hallucinations (87%).

---

### 🛡️ Task 5: Robustness

| Metric | Score |
|--------|-------|
| Average Word Overlap | **0.316** |
| Most Robust Example | Neural Radiance Fields (0.417) |
| Least Robust Example | Knowledge Distillation (0.280) |

> **💡 Insight:** The model shows **limited robustness** and relies on specific lexical cues rather than deep semantic understanding.

---

### 🚀 Task 6: Improvement Method

Switching from **Temperature 1.0 → Greedy** decoding:

| Metric | Before (Temp 1.0) | After (Greedy) | Change |
|--------|-------------------|----------------|--------|
| Hallucination Rate | 53.3% | **0%** | ↓ 53.3% |
| Grounded Rate | 13.3% | **73.3%** | ↑ 60.0% |
| Exact-Match Rate | 10% | **100%** | ↑ 90% |
| Stability | Low | **High** | ↑ |

> **💡 Insight:** Greedy decoding **eliminates hallucination** and **maximizes reproducibility** with the tradeoff of reduced creativity.

---

## 📝 Final Report

The complete report answers all **9 research questions**:

1. **Which decoding method was most reproducible?** → Greedy (100% exact-match)
2. **Which method produced the best titles?** → Greedy (most grounded)
3. **Did the most reproducible method produce the best titles?** → Yes
4. **Which method hallucinated the most?** → Temperature 1.0 (87%)
5. **Did hallucination increase with temperature?** → Yes (0% → 80%)
6. **Was the model robust to paraphrased inputs?** → No (0.316 average similarity)
7. **Did the trust-improvement method help?** → Yes (eliminated hallucination)
8. **Main limitations:** Small dataset, automated labeling, single architecture
9. **Future work:** Larger dataset, more models, human evaluation

---

## 🛠️ Dependencies

```
Python 3.8+
PyTorch 2.0+
Transformers (HuggingFace)
Datasets
Pandas, NumPy
Matplotlib, Seaborn
NLTK
Rouge-Score
Sentence-Transformers

---

## 👤 Author

**Manoj Kumar Anamdasu**
