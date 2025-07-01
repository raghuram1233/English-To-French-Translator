# French to English Neural Machine Translation

A sequence-to-sequence neural machine translation model built with TensorFlow that translates French text to English using an encoder-decoder architecture with attention mechanism.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Training Details](#training-details)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a neural machine translation system that translates French sentences to English. The model uses an encoder-decoder architecture with attention mechanism, built using TensorFlow and Keras. The system preprocesses text data and trains on French-English sentence pairs.

## ✨ Features

- **Encoder-Decoder Architecture**: Bidirectional GRU encoder with attention-based decoder
- **Text Preprocessing**: Comprehensive text normalization and tokenization
- **Attention Mechanism**: Cross-attention layer for improved translation quality
- **Visualization**: Training progress plots and attention weight visualization
- **Flexible Translation**: Support for variable-length input sequences

## 🏗️ Architecture

### Model Components

1. **Encoder**

   - Bidirectional GRU layers
   - Embedding layer for input tokens
   - Text preprocessing and vectorization

2. **Decoder**

   - Unidirectional GRU with attention
   - Cross-attention mechanism
   - Output projection layer

3. **Attention Mechanism**
   - Multi-head attention for encoder-decoder alignment
   - Layer normalization and residual connections

### Text Processing Pipeline

```
Raw Text → Normalization → Tokenization → Vectorization → Model Input
```

## 📊 Dataset

The model is trained on French-English sentence pairs from the `fra.txt` file. The dataset includes:

- **Format**: Tab-separated French-English sentence pairs
- **Preprocessing**: UTF-8 normalization, punctuation handling, and special token insertion
- **Vocabulary**: Limited to 10,000 most frequent tokens per language
- **Split**: 80% training, 20% validation

## 🚀 Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- TensorFlow Text
- NumPy
- Pandas
- Matplotlib

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd French_To_English
```

2. Install required packages:

```bash
pip install tensorflow tensorflow-text numpy pandas matplotlib
```

3. Ensure you have the `fra.txt` dataset file in the project directory.

## 💻 Usage

### Training the Model

1. Open the Jupyter notebook:

```bash
jupyter notebook Main.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the data
   - Build the model architecture
   - Train the model

### Making Translations

```python
# Example translation
french_text = "C'est un jour merveilleux"
translation = model.translate([french_text])
english_text = translation[0].numpy().decode()
print(f"French: {french_text}")
print(f"English: {english_text}")
```

### Batch Translation

```python
french_sentences = [
    "Comment allez-vous?",
    "J'aime beaucoup ce livre",
    "Où est la bibliothèque?"
]
translations = model.translate(french_sentences)
for i, translation in enumerate(translations):
    print(f"French: {french_sentences[i]}")
    print(f"English: {translation.numpy().decode()}")
```

## 📈 Model Performance

### Training Configuration

- **Model Units**: 256 hidden dimensions
- **Vocabulary Size**: 10,000 tokens per language
- **Batch Size**: 64
- **Optimizer**: Adam
- **Loss Function**: Masked Sparse Categorical Crossentropy
- **Early Stopping**: Patience of 3 epochs

## 📁 File Structure

```
French_To_English/
├── Main.ipynb              # Main training notebook
├── Main - Copy.ipynb       # Backup/alternative notebook
├── fra.txt                 # French-English dataset
├── README.md              # This file
└── (generated files)
    ├── model_weights/     # Saved model weights
    └── plots/            # Training visualization plots
```

## 📦 Dependencies

| Package         | Version | Purpose                 |
| --------------- | ------- | ----------------------- |
| TensorFlow      | ≥2.0    | Deep learning framework |
| TensorFlow Text | ≥2.0    | Text preprocessing      |
| NumPy           | ≥1.19   | Numerical operations    |
| Pandas          | ≥1.0    | Data manipulation       |
| Matplotlib      | ≥3.0    | Visualization           |

## 🎯 Training Details

### Data Preprocessing

1. **Text Normalization**: UTF-8 NFKD normalization
2. **Case Handling**: Lowercase conversion
3. **Punctuation**: Space padding around punctuation
4. **Special Tokens**: [START] and [END] token insertion
5. **Vocabulary**: Top 10,000 most frequent words

### Model Training

- **Architecture**: Encoder-decoder with attention
- **Training Strategy**: Teacher forcing during training
- **Inference**: Beam search with temperature control
- **Regularization**: Dropout and early stopping

### Hyperparameters

```python
units = 256                 # Hidden layer size
max_vocab_size = 10000     # Vocabulary size
batch_size = 64            # Training batch size
epochs = 100               # Maximum epochs
patience = 3               # Early stopping patience
```
