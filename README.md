# Transformer from Scratch 

This repository contains a **complete PyTorch implementation** of the **Transformer architecture** from the seminal paper:
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al.

Unlike toy examples, this version is **training-ready**, equipped with advanced features like:

* 🧠 Learned positional embeddings (instead of sinusoidal)
* ✅ Label smoothing for better generalization
* 📉 BLEU score evaluation during validation
* �� Gradient accumulation for stability
* ↻ Greedy decoding
* ⏱ Early stopping and learning rate scheduling

---

## 🚀 Features

### ✅ Model

* Encoder-decoder architecture using:

  * Multi-Head Attention
  * Position-wise Feed-Forward Networks
  * Layer Normalization
* **Learned positional embeddings** (trainable), as allowed by the original paper.
* Token embeddings for both source and target vocabularies.
* Dropout regularization in attention and feedforward layers.

### 💠 Training

* **CrossEntropyLoss with Label Smoothing** (`label_smoothing=0.1`)
* **Gradient Accumulation** for large batch emulation
* **Learning Rate Scheduler**: `ReduceLROnPlateau`
* **Gradient Clipping** to prevent exploding gradients
* **Early Stopping** based on validation loss

### 📈 Evaluation

* Greedy decoding inference
* **BLEU Score** calculation using `nltk` for real translation performance
* Token-level loss and detailed training logs per epoch

---

## 🧩 Directory Structure

```
project/
│
├── transformer.py               # Transformer model with encoder, decoder, attention, FFN        
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
git clone https://github.com/Zaheerkhn/Transformer/edit/main/README.md
cd Transformer

# Set up environment
pip install -r requirements.txt

# Download NLTK data for BLEU
python -c "import nltk; nltk.download('punkt')"
```

---

## 🧪 Training

```bash
python transformer.py
```

Hyperparameters are configurable, including:

| Parameter           | Value                            | Description                                         |
| ------------------- | -------------------------------- | --------------------------------------------------- |
| `n_embed`           | 192                              | Embedding dimension size                           |
| `num_heads`         | 8                                | Number of attention heads                          |
| `num_layer`         | 6                                | Number of Transformer encoder/decoder layers       |
| `dropout`           | 0.2                              | Dropout rate for regularization                    |
| `max_seq_len`       | 48                               | Maximum sequence length supported                  |
| `device`            | `'cuda' if available else 'cpu'` | Device used for training (GPU or CPU)              |
| `num_epochs`        | 10                               | Total number of training epochs                    |
| `patience`          | 3                                | Early stopping patience based on validation loss   |
| `print_every`       | 100                              | Logging frequency (steps)                          |
| `accumulation_steps`| 4                                | Steps to accumulate gradients before update        |
| `lr`                | `3e-4 * accumulation_steps`       | Scaled learning rate with accumulation             |


---

Also shows:

* Validation loss
* BLEU score
* Learning rate updates

---

## 📊 BLEU Score

BLEU is calculated using NLTK:

* We apply tokenization and smoothing
* Scores are averaged across validation set per epoch
* This gives a real sense of translation quality improvement over time

---

## 🧠 Notes

* Learned positional embeddings often outperform sinusoidal ones for many tasks.
* Use higher parameters for better performance.
* The architecture matches the original paper, except:

  * Optimizer is Adam (not Noam), but a scheduler is used
  * Dropout, label smoothing, and clipping are added for real-world usage

---

## 📚 References

* [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* PyTorch documentation
* NLTK BLEU implementation


---

## 🙌 Acknowledgments

Built by [Zaheer Khan](https://github.com/Zaheerkhn) — feel free to connect or raise issues to collaborate or improve this project.
