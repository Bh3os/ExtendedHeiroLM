# BIHieroLM - Egyptian Hieroglyph Prediction Models

## Overview

BIHieroLM is a PyTorch-based implementation of various neural language models designed specifically for Egyptian hieroglyph prediction and restoration. The project aims to aid Egyptologists and researchers in recovering damaged or missing hieroglyphs in ancient texts by leveraging the power of deep learning sequence models.

## Background

Ancient Egyptian hieroglyphic texts often suffer from degradation and damage due to their age. Traditional manual restoration methods are time-consuming and require extensive domain expertise. HieroLM offers an automated approach using neural language models to predict missing hieroglyphs based on their surrounding context.

The models in this repository have been trained on two major datasets:
- **AES**: Ancient Egyptian Sentences corpus
- **Ramses**: A collection of texts from the Ramses II era
- **Combined**: A merged dataset combining unique tokens from both AES and Ramses

## Model Architectures

This implementation now supports multiple LSTM-based architectures:

1. **LSTM** (`lstm`) - Standard unidirectional LSTM (original HieroLM)
2. **LSTM with Attention** (`lstm_attention`) - Unidirectional LSTM with attention mechanism
3. **BiLSTM** (`bilstm`) - Bidirectional LSTM
4. **BiLSTM with Attention** (`bilstm_attention`) - Bidirectional LSTM with attention mechanism

## Usage

### Training Models

For best results, all models should be trained on the combined dataset, which merges unique tokens from both AES and Ramses datasets. Use the `--model_type` parameter to select the architecture:

#### Train Standard LSTM (Original HieroLM):
```bash
python main.py --cuda True --dataset combined --model_type lstm
```

#### Train LSTM with Attention:
```bash
python main.py --cuda True --dataset combined --model_type lstm_attention
```

#### Train Bidirectional LSTM:
```bash
python main.py --cuda True --dataset combined --model_type bilstm
```

#### Train Bidirectional LSTM with Attention:
```bash
python main.py --cuda True --dataset combined --model_type bilstm_attention
```

### Testing Models

To test trained models on the test set:

#### Test Standard LSTM:
```bash
python main.py --cuda True --dataset [aes/ramses/mixed] --mode decode --model_type lstm
```

#### Test LSTM with Attention:
```bash
python main.py --cuda True --dataset [aes/ramses/mixed] --mode decode --model_type lstm_attention
```

#### Test Bidirectional LSTM:
```bash
python main.py --cuda True --dataset [aes/ramses/mixed] --mode decode --model_type bilstm
```

#### Test Bidirectional LSTM with Attention:
```bash
python main.py --cuda True --dataset [aes/ramses/mixed] --mode decode --model_type bilstm_attention
```

### Real-time Interaction

To interact with trained models in real time:

#### Interact with Standard LSTM:
```bash
python main.py --cuda True --dataset [aes/ramses/mixed] --mode realtime --model_type lstm
```

#### Interact with LSTM with Attention:
```bash
python main.py --cuda True --dataset [aes/ramses/mixed] --mode realtime --model_type lstm_attention
```

#### Interact with Bidirectional LSTM:
```bash
python main.py --cuda True --dataset [aes/ramses/mixed] --mode realtime --model_type bilstm
```

#### Interact with Bidirectional LSTM with Attention:
```bash
python main.py --cuda True --dataset [aes/ramses/mixed] --mode realtime --model_type bilstm_attention
```

### Multi-shot Evaluation

For multi-shot accuracy evaluation:

```bash
python multi.py --cuda True --dataset [aes/ramses/mixed] --model_type [lstm/lstm_attention/bilstm/bilstm_attention]
```

## Model Configuration

All models share the same hyperparameters and configuration options:

- `--embed_size`: Embedding dimension (default: 1024)
- `--hidden_size`: Hidden state dimension (default: 1024)
- `--dropout`: Dropout rate (default: 0)
- `--lr`: Learning rate (default: 1e-3)
- `--max_epoch`: Maximum training epochs (default: 50)
- `--train_batch_size`: Training batch size (default: 32)

## Model Architecture Details

All models in this repository are based on LSTM architectures with various enhancements. Each model inherits from a base `BaseHieroLM` class and implements its specific architecture.

### Standard LSTM (`lstm`)
- **Architecture**: Unidirectional LSTM encoder
- **Processing**: Processes tokens sequentially from left to right
- **Output**: Simple linear projection from hidden states to vocabulary distribution
- **Best for**: Basic sequence prediction with minimal computational requirements
- **Implementation**: `HieroLM` class in `models.py`

### LSTM with Attention (`lstm_attention`)
- **Architecture**: Unidirectional LSTM encoder with attention mechanism
- **Processing**: Processes tokens sequentially and computes attention weights across all positions
- **Attention**: Implements an attention mechanism that attends to all encoder hidden states
- **Output**: Concatenates attention context with hidden states before projection to vocabulary
- **Best for**: Capturing important context from different parts of the input sequence
- **Implementation**: `HieroLMAttention` class in `models.py`

### Bidirectional LSTM (`bilstm`)
- **Architecture**: Bidirectional LSTM encoder
- **Processing**: Processes tokens in both forward and backward directions simultaneously
- **Output**: Concatenates forward and backward hidden states before projection
- **Output dimension**: Uses double the hidden dimension in the final projection layer
- **Best for**: Capturing context from both preceding and following tokens
- **Implementation**: `BiHieroLM` class in `models.py`

### Bidirectional LSTM with Attention (`bilstm_attention`)
- **Architecture**: Bidirectional LSTM encoder with attention mechanism
- **Processing**: Processes tokens in both directions and applies attention mechanism
- **Attention**: Applies attention over the combined bidirectional hidden states
- **Output**: Combines bidirectional encoding with attention context before projection
- **Best for**: Most complex model, ideal for capturing long-distance dependencies in both directions
- **Implementation**: `BiHieroLMAttention` class in `models.py`

### Attention Mechanism
The attention mechanism implemented in this codebase follows the general formulation:

1. Calculate attention scores using hidden states
2. Apply softmax to get attention weights
3. Create context vector by weighted sum of hidden states
4. Concatenate with original hidden state before final projection

This allows the model to focus on different parts of the input sequence when making predictions.

## Repository Structure

```
HieroLM-main/
│
├── models.py                  # Model architecture definitions (LSTM, BiLSTM, Attention models)
├── main.py                    # Main script for training, evaluation, and interactive testing
├── multi.py                   # Multi-shot evaluation for predicting consecutive hieroglyphs
├── parse.py                   # Command-line argument parsing and configuration
├── utils.py                   # Utility functions for data processing and evaluation
├── vocab.py                   # Vocabulary handling and processing
├── train_all_models.py        # Script to train and evaluate all model variants
├── test_models.py             # Utility scripts for comprehensive model testing
├── compare_models.py          # Tools for comparing performance across model architectures
├── verify_models.py           # Verification scripts for model integrity
├── create_combined_vocab.py   # Script to create combined vocabulary from multiple datasets
│
├── data/                      # Contains all datasets used for training and evaluation
│   ├── aes/                   # Ancient Egyptian Sentences corpus
│   ├── ramses/                # Ramses II era texts corpus
│   ├── mixed/                 # Mixed dataset (historical combination)
│   └── combined/              # Combined dataset merging AES and Ramses
│       ├── train.txt          # Training data
│       ├── val.txt            # Validation data
│       ├── test.txt           # Test data
│       └── vocab.json         # Vocabulary file with unique tokens
│
├── saved_models/              # Directory for storing trained model checkpoints
│
└── runs/                      # TensorBoard log directory for training visualization
    ├── lstm/                  # Standard LSTM logs
    ├── lstm_attention/        # LSTM with attention logs  
    ├── bilstm/                # Bidirectional LSTM logs
    └── bilstm_attention/      # Bidirectional LSTM with attention logs
```

### Core Files

- **models.py**: Defines all model architectures including the base HieroLM class and its variants (attention-based and bidirectional). Provides factory functions for model instantiation and loading.

- **main.py**: The primary entry point for training, testing, and running models in real-time interactive mode. Handles data loading, model training loops, and evaluation metrics.

- **multi.py**: Implements multi-shot evaluation metrics for assessing model performance on predicting sequences of consecutive hieroglyphs rather than just the next token.

- **parse.py**: Command-line argument parser that defines all hyperparameters, model settings, and runtime configurations.

- **utils.py**: Contains utility functions for data processing, batch creation, and other helper functions used throughout the codebase.

- **vocab.py**: Manages vocabulary creation, token-to-index mapping, and all text processing functionality required for model input/output.

### Training and Evaluation

- **train_all_models.py**: A convenience script to sequentially train and evaluate all model variants with the same dataset and hyperparameters.

- **test_models.py**: Provides comprehensive testing utilities to validate model performance across different metrics and test sets.

- **compare_models.py**: Tools for comparative analysis between different model architectures, generating performance comparison reports.

- **verify_models.py**: Validates model integrity and ensures proper functioning of all model components.

### Data Management

- **create_combined_vocab.py**: Script to build a combined vocabulary by extracting unique tokens from multiple source datasets.

## Requirements and Setup

### Dependencies

```
torch>=1.7.0
numpy>=1.19.0
tqdm>=4.45.0
scikit-learn>=0.23.0
tensorboard>=2.3.0
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/HieroLM.git
cd HieroLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create the combined dataset vocabulary (if not already present):
```bash
python create_combined_vocab.py
```

### CUDA Support

All models support GPU acceleration using CUDA. To enable CUDA, use the `--cuda True` parameter when running any script. The code automatically falls back to CPU if CUDA is not available.

## Model Naming Convention

Trained models are saved with the following naming pattern:
```
saved_models/{embed_size}_{hidden_size}_{dropout}_{dataset}_{model_type}_{model_save_path}
```

Example:
- `saved_models/1024_1024_0_combined_lstm_model.bin`
- `saved_models/1024_1024_0_combined_lstm_attention_model.bin`
- `saved_models/1024_1024_0_combined_bilstm_model.bin`
- `saved_models/1024_1024_0_combined_bilstm_attention_model.bin`

## Training All Models At Once

You can train, test, and evaluate all model variants with a single command using the `train_all_models.py` script. For optimal results, use the combined dataset:

```bash
python train_all_models.py combined [cuda] [epochs]
```

Example:
```bash
python train_all_models.py combined True 20
```

This will sequentially:
1. Train all four model variants
2. Test each trained model
3. Perform multi-shot evaluation
4. Provide a summary of results

## Applications

### Hieroglyph Restoration
HieroLM models can be used to automatically restore damaged or missing hieroglyphs in ancient Egyptian texts:
- Input a partial text with missing hieroglyphs
- The model predicts the most likely hieroglyph(s) to fill the gaps
- Useful for archaeological research and text reconstruction

### Text Completion
Interactive text completion for researchers working with hieroglyphic texts:
- As researchers transcribe texts, the model suggests the next likely hieroglyphs
- Can speed up transcription work and provide options for ambiguous cases

### Educational Tool
The models can serve as educational tools for students of ancient Egyptian language:
- Practice hieroglyphic writing with intelligent feedback
- Learn common patterns and sequences in hieroglyphic texts

### Research on Ancient Egyptian Language
These models provide quantitative insights into hieroglyphic language patterns:
- Frequency analysis of hieroglyph sequences
- Identification of common linguistic structures
- Statistical analysis of hieroglyphic language usage

## Citation

If you use HieroLM in your research, please cite our paper:

```
@inproceedings{hierolm2025,
  title={HieroLM: Egyptian Hieroglyph Recovery with Next Word Prediction Language Model},
  author={[Author Names]},
  booktitle={The 9th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2025)},
  year={2025},
  publisher={Association for Computational Linguistics}
}
```

## Datasets and Data Format

### Available Datasets

1. **AES (Ancient Egyptian Sentences)**
   - Contains hieroglyphic texts from various periods of ancient Egypt
   - Primarily consists of narrative and administrative texts
   - Format: One sentence per line, tokens separated by spaces

2. **Ramses**
   - Texts mainly from the Ramses II era and related periods
   - Contains royal decrees, religious texts, and some administrative documents
   - Format: One sentence per line, tokens separated by spaces

3. **Combined**
   - Merges the unique vocabularies of AES and Ramses datasets
   - Provides the most comprehensive coverage of hieroglyphic tokens
   - Recommended dataset for training all models for best performance

### Data Format

All datasets follow the same format:
- Each line represents a complete sentence or text fragment
- Tokens are separated by spaces
- Each token corresponds to a single hieroglyph or hieroglyph grouping
- Example: `jw j r s m grH` (I came to it at night)

### Vocabulary Files

The vocabulary files (vocab.json) map each token to a unique index:
- Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`
- All hieroglyphic tokens found in the dataset
- The combined vocabulary includes all unique tokens from both AES and Ramses datasets

## Performance Evaluation

Models are evaluated using several metrics:

### Next Token Accuracy
Measures the model's ability to correctly predict the next hieroglyph given a sequence of previous hieroglyphs.

### Multi-shot Accuracy
Evaluates the model's ability to predict consecutive hieroglyphs in sequence:
- 1-shot: Predict the next hieroglyph
- 2-shot: Predict the next 2 hieroglyphs in sequence
- 3-shot: Predict the next 3 hieroglyphs in sequence
- 4-shot: Predict the next 4 hieroglyphs in sequence

### Perplexity
Measures the model's confidence in its predictions; lower perplexity indicates better modeling of the hieroglyphic language.

### F1 Score
Balances precision and recall in token prediction, particularly useful for evaluating performance on less frequent hieroglyphs.

## Future Work

Several directions for future improvements and extensions include:

1. **Transformer-based Models**: Implementing transformer architectures for hieroglyph prediction, which may capture even more complex dependencies.

2. **Visual-Textual Integration**: Incorporating visual features of hieroglyphs into the prediction models to handle the pictographic nature of the writing system.

3. **Morphological Analysis**: Adding explicit modeling of hieroglyphic morphology to improve prediction accuracy.

4. **Pre-training on Larger Corpora**: Developing pre-trained models on an expanded corpus of hieroglyphic texts.

5. **Interactive Tools for Egyptologists**: Developing more sophisticated user interfaces for practical application in archaeological research.

## Contributing

Contributions to the HieroLM project are welcome! Please feel free to submit pull requests or open issues to improve the models, add features, or fix bugs.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
