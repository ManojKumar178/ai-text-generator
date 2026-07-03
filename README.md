Trustworthy AI Text Generator
Evaluating the trustworthiness of a simple text generation model (t5-small) for academic title generation.

https://colab.research.google.com/assets/colab-badge.svg

📋 Overview
This project evaluates whether a small generative AI model (t5-small) can produce useful academic paper titles while remaining reproducible, grounded in source abstracts, and robust to paraphrased inputs. We fine-tune t5-small on 70 abstract-title pairs and evaluate across five decoding methods to assess trustworthiness properties.

Main Research Question: Can a small generative AI model produce useful text while remaining reproducible, grounded, and robust?

🎯 Key Features
Fine-tunes t5-small (~60M parameters) for abstract-to-title generation

Evaluates 5 decoding methods: Greedy, Temperature (0.7, 1.0, 1.3), Top-k (k=10)

Tests 3 trustworthiness properties:

Reproducibility: Does the model produce consistent outputs?

Grounding: Are generated titles supported by input text?

Robustness: Does the model handle paraphrased inputs?

📊 Dataset
100 real abstract-title pairs from computer science and machine learning papers

Split: 70 training, 15 validation, 15 trust-evaluation (never used for training)

Each example contains: abstract (input) and title (target)

All titles ≤ 32 tokens (compatible with t5-small max_length)

🚀 Getting Started
Option 1: Run on Google Colab (Recommended)
https://colab.research.google.com/assets/colab-badge.svg

Click the badge above to open the notebook in Colab and run all experiments.

Option 2: Run Locally
Clone the repository

bash
git clone https://github.com/ManojKumar178/ai-text-generator.git
cd ai-text-generator
Install dependencies

bash
pip install -r requirements.txt
Run the notebook

bash
jupyter notebook code/trust_ai_assignment.ipynb
Option 3: Use the Code Files
python
# Example: Generate a title from an abstract
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def generate_title(abstract):
    input_text = "summarize: " + abstract
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=256)
    outputs = model.generate(inputs.input_ids, max_length=64, min_length=2, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
abstract = "This study proposes joint models for intent detection and slot filling..."
title = generate_title(abstract)
print(title)
📁 Repository Structure
text
ai-text-generator/
├── code/
│   └── trust_ai_assignment.ipynb   # Main Colab notebook
├── dataset/
│   ├── train.csv                   # 70 training examples
│   ├── val.csv                     # 15 validation examples
│   └── eval.csv                    # 15 trust-evaluation examples
├── output/
│   ├── task1_generated_titles.csv  # Task 1: Basic generation
│   ├── task2_generated_titles.csv  # Task 2: All decoding methods
│   ├── task3_reproducibility.csv   # Task 3: Reproducibility results
│   ├── task4_hallucination.csv     # Task 4: Hallucination analysis
│   ├── task5_robustness.csv        # Task 5: Robustness results
│   └── loss_curves.png             # Training/validation loss curves
├── README.md
└── requirements.txt
📈 Key Results
Task 3: Reproducibility
Decoding Method	Exact-Match Rate	Stability
Greedy	100%	High
Temperature 0.7	13%	Low
Temperature 1.0	10%	Low
Temperature 1.3	10%	Low
Top-k (k=10)	10%	Low
Conclusion: Greedy decoding is the most reproducible method (100% exact-match) because it is deterministic.

Task 4: Hallucination
Decoding Method	Hallucination Rate
Greedy	0%
Temperature 0.7	13%
Temperature 1.0	87%
Temperature 1.3	80%
Top-k (k=10)	47%
Conclusion: Hallucination increases with temperature. Greedy decoding eliminates hallucination entirely.

Task 5: Robustness
Average Word Overlap: 0.316 (low robustness)

Most Robust Example: Neural Radiance Fields (0.417 similarity)

Least Robust Example: Knowledge Distillation (0.280 similarity)

Conclusion: The model relies on specific lexical cues rather than deep semantic understanding.

Task 6: Improvement Method
Switching from Temperature 1.0 to Greedy decoding:

Hallucination: 53.3% → 0% (↓ 53.3%)

Grounded Rate: 13.3% → 73.3% (↑ 60.0%)

Exact-Match: 10% → 100% (↑ 90%)

Tradeoff: Reduced creativity for improved reliability.

🛠️ Dependencies
Python 3.8+

PyTorch

Transformers (HuggingFace)

Datasets

Pandas, NumPy

Matplotlib, Seaborn

NLTK

Rouge-Score

Sentence-Transformers

Full list in requirements.txt.

📝 Final Report
The complete report is available in the repository as FINAL_REPORT.md or report.pdf. It answers all research questions and provides detailed analysis of each task.

Research Questions Answered:

Which decoding method was most reproducible? → Greedy

Which method produced the best titles? → Greedy (most grounded)

Did the most reproducible method produce the best titles? → Yes

Which method hallucinated the most? → Temperature 1.0 (87%)

Did hallucination increase with temperature? → Yes (0% → 80%)

Was the model robust to paraphrased inputs? → No (0.316 average similarity)

Did the trust-improvement method help? → Yes (eliminated hallucination)

Main limitations: Small dataset, automated labeling, single architecture

Future work: Larger dataset, more models, human evaluation

📄 License
This project is for educational and research purposes.

👤 Author
Manoj Kumar Anamdasu

GitHub: @ManojKumar178

