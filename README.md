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
  - finetuning-lora-kaggle.ipynb - Jupyter notebook for using the LoRA fine-tuning code on kaggle. This is the main method to use this repository. Detailed instructions of how it works are included below.

- **report/**
  - Deep_Learning_Project_2.pdf - Detailed project report

- **results/**
  - Has the logs for the best results.

- **src/** - Source code
  - evaluate.py - Model evaluation utilities
  - model.py - Model definition
  - train.py - Training code

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
```

## Using the Kaggle Notebook

The `finetuning-lora-kaggle.ipynb` notebook provides a convenient way to use the LoRA fine-tuning code on Kaggle. It streamlines the process of training the RoBERTa model with LoRA on the AGNEWS dataset and running inference on new data.

The notebook includes the following steps:

1. **Environment Setup**: Install dependencies and clone the repository.
    ```python
    # Install required packages
    !pip install transformers datasets evaluate accelerate peft trl bitsandbytes
    
    # Clone the repository
    !git clone https://github.com/jessenb16/Finetuning-with-LoRA.git
    %cd Finetuning-with-LoRA
    ```

2. **Path Configuration**: Set up input and output paths.
    ```python
    pkl_file = "/kaggle/input/deep-learning-spring-2025-project-2/test_unlabelled.pkl"
    output_direct = '/kaggle/working/output'
    os.makedirs(output_direct, exist_ok=True)
    ```

3. **LoRA Configuration**: Define the LoRA parameters.
    ```python
    loraConfig = LoraConfig(
        r=22,                    # Rank of the LoRA layers
        lora_alpha=44,           # Scaling factor for LoRA
        lora_dropout=0.1,        # Dropout rate for LoRA layers
        bias="none",             # Bias handling in LoRA layers
        task_type="SEQ_CLS",     # Task type for sequence classification
        target_modules=['value'] # Which modules to apply LoRA to
    )
    ```

4. **Training Setup**: Configure training parameters.
    ```python
    training_args = TrainingArguments(
        output_dir = output_direct,
        eval_strategy = "steps",
        learning_rate = 5e-5,
        max_steps = 200,
        warmup_steps = 100,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 64,
        # Additional parameters...
    )
    ```

5. **Model Creation**: Initialize the LoRA-adapted model.
    ```python
    model, id2label = setup_lora_model(loraConfig)
    ```

6. **Dataset Preparation**: Load and preprocess the data.
    ```python
    train_dataset, eval_dataset, data_collator = get_datasets()
    ```

7. **Training**: Train the model with the specified configuration.
    ```python
    # Train the model
    train(model, train_dataset, eval_dataset, data_collator, training_args, id2label)
    ```

8. **Inference**: Run predictions on unlabeled test data.
    ```python
    # Load the test dataset and run inference
    unlabelled_data = get_unlabelled_dataset(pkl_file, tokenizer)
    run_inference(model, unlabelled_data, output_direct, data_collator)
    ```
9. **Notes**: There is some randomness involved and the results will not be consistent for every run even with the same parameters. The results currently shown in the finetuning-lora-kaggle.ipynb are not the best results. Please look in results folder for the best results.
