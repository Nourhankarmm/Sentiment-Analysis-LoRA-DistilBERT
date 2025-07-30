# 💬 Sentiment Analysis using DistilBERT + LoRA (PEFT)

Fine-tuning a lightweight transformer (DistilBERT) using **LoRA (Low-Rank Adaptation)** for efficient and accurate sentiment classification on text data.

---

## 🚀 Project Overview

This project tackles **sentiment analysis** by fine-tuning a pre-trained `DistilBERT` model using LoRA — a **Parameter-Efficient Fine-Tuning (PEFT)** method that trains a small number of additional weights while keeping most of the original model frozen.  
This allows fast, memory-efficient training even on limited resources.

---

## 🧠 What is LoRA?

**LoRA (Low-Rank Adaptation)** freezes the original model weights and injects low-rank trainable matrices into attention layers.  
It reduces the number of trainable parameters dramatically and makes large models usable on smaller hardware.

> ✅ Ideal for training on laptops, Google Colab, or limited GPUs.

---

## 📁 Dataset

The dataset consists of text samples labeled with their sentiment:  
- `positive`  
- `negative`

<pre>
dataset/
├── train.csv
├── test.csv
└── sample_submission.csv
</pre>

Each row typically includes:
- `text`: the input sentence or review  
- `label`: the sentiment (0 = negative, 1 = positive)

---

## 🏗️ Project Structure

<pre>
├── notebooks/
│   └── sentiment_lora_distilbert.ipynb
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── assets/
│   └── confusion_matrix.png
│
├── requirements.txt
└── README.md
</pre>

---

## ⚙️ Installation

git clone https://github.com/Nourhankarmm/sentiment-lora-distilbert.git
cd sentiment-lora-distilbert
pip install -r requirements.txt

## 📊 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
Confusion Matrix (visualized in assets/confusion_matrix.png)


## 📦 Tech Stack
Python
Hugging Face Transformers
🤗 PEFT (LoRA integration)
PyTorch
Scikit-learn
Pandas / Matplotlib

