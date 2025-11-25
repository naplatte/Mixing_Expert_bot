# Mixing Expert Bot

This project is a bot designed to intelligently mix various elements based on user specifications. It leverages advanced algorithms to provide optimal mixing solutions and enhance user experience. 

## Features
- Intelligent mixing algorithms
- User-friendly interface
- Customization options

## Architecture Overview

### DesExpert Model

The `DesExpert` model is a description-based expert that uses BERT for bot detection. Here's how the input pipeline and training process works:

#### Input Pipeline

1. **Preprocessing** (`Des_preprocess` in `dataset.py`):
   - Extracts user description strings from the raw data
   - Handles missing descriptions by replacing them with `'None'`
   - Returns a numpy array of description strings

2. **Tokenization** (`DescriptionDataset` in `train_des_experts.py`):
   - Uses `BertTokenizer` from Hugging Face Transformers
   - Converts each description string to `input_ids` and `attention_mask` tensors
   - Pads/truncates sequences to `max_length=128`
   - Returns PyTorch tensors ready for batching

3. **DataLoader**:
   - Batches tokenized data with `batch_size=32`
   - Provides `input_ids` and `attention_mask` tensors to the model

#### BERT Fine-Tuning

The `DesExpert` model **fine-tunes BERT** during training:

- **Model Architecture** (`DesExpert` in `model.py`):
  - Loads `bert-base-uncased` as the encoder
  - Adds an MLP network (768 → 256 → 128 → 64) for expert representation
  - Adds a classifier head (64 → 32 → 1) with sigmoid activation for bot probability

- **Training Process**:
  - Uses `BCELoss` (Binary Cross-Entropy Loss) for binary classification
  - Optimizes with `AdamW` (learning rate: 2e-5)
  - Trains both BERT and the classifier layers **end-to-end** for the bot classification task

#### Data Flow Summary

```
Description strings (Des_preprocess)
        ↓
BertTokenizer (max_length=128)
        ↓
input_ids + attention_mask tensors
        ↓
DataLoader (batch_size=32)
        ↓
DesExpert (BERT + MLP + Classifier)
        ↓
Bot probability output (0-1)
```

## Getting Started
To get started with the Mixing Expert Bot, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/naplatte/Mixing_Expert_bot.git
cd Mixing_Expert_bot
```

## Contribution
Feel free to contribute to the project by submitting pull requests and reporting issues!