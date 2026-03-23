# 🧠 LSTM-Based Text Generation (Generative AI Task)

This project implements a **text generation model using LSTM (Long Short-Term Memory)** networks trained on **Shakespeare’s Complete Works**.  
The model learns language patterns and generates coherent text based on a given seed input.

---

## 🚀 Overview

The goal of this project is to:
- Build a text generation system using deep learning
- Understand sequence modeling with LSTMs
- Generate human-like text using trained models

---

## 📂 Dataset

- **Source:** Project Gutenberg  
- **Dataset:** Shakespeare's Complete Works  
- **Link:** https://www.gutenberg.org/files/100/100-0.txt  

The dataset is automatically downloaded using the script.

---

## ⚙️ Features

- ✔ Data preprocessing (cleaning, tokenization)
- ✔ Sequence generation for training
- ✔ Baseline LSTM model
- ✔ Deep stacked LSTM model (bonus)
- ✔ Early stopping & checkpointing
- ✔ Temperature-based text generation
- ✔ Model comparison report

---

## 🏗️ Model Architecture

### 🔹 Baseline Model
```
Embedding → LSTM → Dropout → Dense (Softmax)
```

### 🔹 Deep Model (Bonus)
```
Embedding → LSTM → Dropout
          → LSTM → Dropout
          → LSTM → Dropout
          → Dense (Softmax)
```

---

## 🧪 Training Details

- Loss Function: `categorical_crossentropy`
- Optimizer: `Adam`
- Sequence Length: `40`
- Batch Size: `128`
- Epochs: `50` (with Early Stopping)

---

## ✨ Text Generation

The model generates text using:
- Seed input (starting phrase)
- Iterative next-word prediction
- Temperature sampling for creativity control

### 🌡️ Temperature Effects:
- `0.3` → Conservative, repetitive
- `0.8` → Balanced (recommended)
- `1.3` → Creative but less coherent

---

## 📝 Sample Output

**Seed:**  
`to be or not to be`

**Generated Text:**  
```
to be or not to be that is the heart of the world and the king shall speak of love and honour in the night ...
```

---

## 📊 Model Comparison

| Metric              | Baseline Model | Deep Model |
|--------------------|---------------|------------|
| Training Speed     | Fast          | Slower     |
| Accuracy           | Good          | Better     |
| Coherence          | Moderate      | Higher     |
| Complexity         | Low           | High       |

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python lstm_text_generation.py
```

---

## 📁 Project Structure

```
lstm-text-generation/
│
├── lstm_text_generation.py
├── requirements.txt
├── README.md
├── outputs.txt
└── checkpoints/
```

---

## 💡 Key Learnings

- Sequence modeling using LSTM
- Handling large text datasets
- Tokenization and vocabulary control
- Importance of hyperparameters like sequence length
- Trade-offs between shallow vs deep architectures

---

## 🎯 Future Improvements

- Use GRU or Transformer models
- Train on larger datasets
- Add web interface for live text generation
- Fine-tune pretrained language models

---

## 👨‍💻 Author

**AI/ML Engineer (B.Tech AIML Student)**  
- Strong focus on Machine Learning & Generative AI  
- Experience with NLP, LLMs, and Deep Learning  

---

## ⭐ Conclusion

This project demonstrates a complete pipeline for:
- Data preprocessing  
- Model building  
- Training  
- Text generation  

It goes beyond the basic requirements by including:
- Advanced sampling techniques  
- Multiple architectures  
- Comparative analysis  
