🤖 Trustworthy AI Text Generator
Evaluating the trustworthiness of a simple text generation model (t5-small) for academic title generation.

https://colab.research.google.com/assets/colab-badge.svg
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/PyTorch-2.0+-red.svg
https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg
https://img.shields.io/badge/License-MIT-green.svg
https://img.shields.io/badge/PRs-welcome-brightgreen.svg

📋 Overview
Can a small AI model generate useful, reproducible, and trustworthy text? This project explores that question by fine-tuning t5-small (~60M parameters) to generate academic paper titles from abstracts.

🎯 Main Research Question
Can a small generative AI model produce useful text while remaining reproducible, grounded, and robust?

We evaluate 3 critical trustworthiness properties:

Property	Question
🔁 Reproducibility	Does the model produce the same output for the same input?
📍 Grounding	Are generated titles supported by the input text?
🛡️ Robustness	Does the model handle paraphrased inputs correctly?
✨ Key Features
<table> <tr> <td width="33%">
🎯 5 Decoding Methods
Greedy

Temperature (0.7, 1.0, 1.3)

Top-k (k=10)

</td> <td width="33%">
📊 3 Trust Tests
Reproducibility (10 runs each)

Hallucination detection

Robustness to paraphrasing

</td> <td width="33%">
🧠 t5-small
~60M parameters

Fine-tuned on 70 examples

10 epochs training

</td> </tr> </table>
📊 Dataset
<details> <summary><b>Click to expand dataset details</b></summary>
Split	Size	Purpose
Training	70 (70%)	Model fine-tuning
Validation	15 (15%)	Hyperparameter tuning
Trust-Evaluation	15 (15%)	NEVER used for training
Data Statistics:

Average abstract length: ~120 words

Average title length: 7.3 words

Title length range: 5–11 words

All titles ≤ 32 tokens ✅

</details>
🚀 Getting Started
Option 1: Run on Google Colab (Recommended)
https://colab.research.google.com/assets/colab-badge.svg

Click the badge above and run the notebook in one click! 🎉

Option 2: Run Locally
bash
# Clone the repository
git clone https://github.com/ManojKumar178/ai-text-generator.git
cd ai-text-generator

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook code/trust_ai_assignment.ipynb
Option 3: Quick Code Snippet
python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Generate title
def generate_title(abstract):
    input_text = "summarize: " + abstract
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=256)
    outputs = model.generate(inputs.input_ids, max_length=64, min_length=2, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Try it!
title = generate_title("This study proposes joint models for intent detection...")
print(title)
📁 Repository Structure
text
ai-text-generator/
├── 📁 code/
│   └── trust_ai_assignment.ipynb   # Main Colab notebook
├── 📁 dataset/
│   ├── train.csv                   # 70 training examples
│   ├── val.csv                     # 15 validation examples
│   └── eval.csv                    # 15 trust-evaluation examples
├── 📁 output/
│   ├── task1_generated_titles.csv  # Task 1: Basic generation
│   ├── task2_generated_titles.csv  # Task 2: All decoding methods
│   ├── task3_reproducibility.csv   # Task 3: Reproducibility results
│   ├── task4_hallucination.csv     # Task 4: Hallucination analysis
│   ├── task5_robustness.csv        # Task 5: Robustness results
│   └── loss_curves.png             # Training/validation loss curves
├── README.md
└── requirements.txt
📈 Key Results
🔁 Task 3: Reproducibility
Decoding Method	Exact-Match Rate	Near-Duplicate Rate	Stability
🥇 Greedy	100%	100%	High 🔥
Temperature 0.7	13%	33%	Low
Temperature 1.0	10%	15%	Low
Temperature 1.3	10%	10%	Low
Top-k (k=10)	10%	14%	Low
💡 Insight: Greedy decoding is 100% reproducible because it's deterministic!

📍 Task 4: Hallucination
Decoding Method	Grounded Rate	Partially Grounded	Hallucination Rate
🥇 Greedy	73%	27%	0% ✅
Temperature 0.7	53%	33%	13%
Temperature 1.0	0%	13%	87% ❌
Temperature 1.3	7%	13%	80%
Top-k (k=10)	20%	33%	47%
💡 Insight: Hallucination increases with temperature. Greedy = 0% hallucination! 🎯

🛡️ Task 5: Robustness
Metric	Score
Average Word Overlap	0.316
Most Robust Example	Neural Radiance Fields (0.417)
Least Robust Example	Knowledge Distillation (0.280)
💡 Insight: The model shows limited robustness and relies on specific lexical cues rather than semantic understanding.

🚀 Task 6: Improvement Method
Switching from Temperature 1.0 → Greedy decoding:

Metric	Before (Temp 1.0)	After (Greedy)	Improvement
Hallucination Rate	53.3%	0%	↓ 53.3%
Grounded Rate	13.3%	73.3%	↑ 60.0%
Exact-Match Rate	10%	100%	↑ 90%
💡 Insight: Greedy decoding eliminates hallucination and maximizes reproducibility - a small price for reduced creativity!

📝 Final Report
The complete report answers 9 research questions:

<details> <summary><b>Click to see all questions answered</b></summary>
Which decoding method was most reproducible? → Greedy (100% exact-match)

Which method produced the best titles? → Greedy (most grounded)

Did the most reproducible method produce the best titles? → Yes

Which method hallucinated the most? → Temperature 1.0 (87%)

Did hallucination increase with temperature? → Yes (0% → 80%)

Was the model robust to paraphrased inputs? → No (0.316 average similarity)

Did the trust-improvement method help? → Yes (eliminated hallucination)

Main limitations: Small dataset, automated labeling, single architecture

Future work: Larger dataset, more models, human evaluation

</details>
🛠️ Dependencies
<details> <summary><b>Click to see full list</b></summary>
Python 3.8+

PyTorch 2.0+

Transformers (HuggingFace)

Datasets

Pandas, NumPy

Matplotlib, Seaborn

NLTK

Rouge-Score

Sentence-Transformers

Full list in requirements.txt

</details>
🤝 Contributing
Contributions are welcome! Feel free to:

⭐ Star this repository

🐛 Report bugs

💡 Suggest improvements

🔀 Submit pull requests

📄 License
This project is for educational and research purposes only. Use responsibly!

👤 Author
Manoj Kumar Anamdasu
