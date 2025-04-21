# Finetuning with LoRA

This repository contains code and experiments for fine-tuning models using Low-Rank Adaptation (LoRA). The project focuses on optimizing a BERT architecture (specifically RoBERTa) for the AGNEWS text classification dataset under the constraint of using no more than 1 million trainable parameters.

LoRA (Low-Rank Adaptation) allows for efficient fine-tuning by freezing the pre-trained model weights and injecting trainable rank decomposition matrices into each layer. This approach dramatically reduces the number of trainable parameters while maintaining model performance. In this implementation, each frozen weight matrix in RoBERTa is perturbed by a trainable low-rank matrix, with configurable rank and perturbation strength parameters.

The experiments explore various LoRA configurations to identify the optimal settings for maximizing accuracy within the parameter budget constraint.

## Model Details

| **Parameter** | **Value** |
|---------------|-----------|
| **LoRA Configuration** | |
| Base Model | `roberta-base` |
| Adapter Scope | `value` |
| Rank (r) | 22 |
| Scaling Factor (α) | 44 |
| Dropout | 0.10 |
| Trainable Parameters | 999,172 |
| **Training Parameters** | |
| Optimizer | AdamW (`adamw_torch`) |
| Learning Rate | 5×10⁻⁵ |
| Warmup Steps | 100 |
| Total Steps | 1,200 |
| Eval Strategy | steps (every 100) |
| Train Batch Size | 16 |
| Eval Batch Size | 64 |
| DataLoader Workers | 4 |
| **Final Performance** | |
| Training Loss | 0.2993 |
| Validation Loss | 0.2965 |
| Peak Validation Accuracy | **91.41%** |

## Project Structure

- **experiments/** - Contains experiment results organized by parameter:
  - `a/` - Experiments with different alpha values
  - `additional/` - Additional experiments
  - `dropout/` - Experiments with different dropout rates
  - `lr/` - Learning rate experiments
  - `modules/` - Experiments with different target modules
  - `r/` - Experiments with different rank values

- **notebooks/**
  - LoRA_visualization_notebook.ipynb - Jupyter notebook for visualizing experiment results

- **report/**
  - Deep_Learning_Project_2.pdf - Detailed project report

- **src/** - Source code
  - evaluate.py - Model evaluation utilities
  - model.py - Model definition
  - train.py - Training code
  - utils.py - Utility functions

- **visualizations/** - Generated visualizations
  - Five experiment visualizations showing comparative performance

## Experiments

The project includes five main experiments:

1. **Experiment 1**: Peak Validation Accuracy by Adapter Scope
2. **Experiment 2**: Eval Accuracy vs. Step for Different Dropout Rates
3. **Experiment 3**: Eval Accuracy vs. Step for Alpha (r*0.5, r*1, r*2)
4. **Experiment 4**: Eval Accuracy vs. Step for Different r Values
5. **Experiment 5**: Eval Accuracy vs. Step for Different Learning Rates

## Installation

To install the required packages:

```bash
pip install -r requirements.txt
