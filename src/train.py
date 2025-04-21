from datasets import load_dataset
from transformers import RobertaTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score
from src.model import setup_lora_model


def load_data():
    """
    Load and preprocess the AG News dataset for sequence classification.
    Returns:
        tokenized_dataset: Preprocessed dataset ready for training.
        data_collator: Data collator for batching the data.
    """
    # Load the AG News dataset
    base_model = 'roberta-base'
    dataset = load_dataset('ag_news', split='train')
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    
    def preprocess(examples):
        # Ensure return of dictionary with input_ids, attention_mask, etc.
        return tokenizer(examples['text'], truncation=True, padding=True, return_tensors=None)
    
    # Make sure preprocess handles batched inputs properly
    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
    
    # Make sure the dataset has 'labels' column (not 'label')
    if "label" in tokenized_dataset.column_names and "labels" not in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    # Check that input_ids are present in the dataset
    print(f"Tokenized dataset features: {tokenized_dataset.column_names}")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    return tokenized_dataset, data_collator

def split_dataset(tokenized_dataset):
    """
    Split the tokenized dataset into training and evaluation sets.
    Args:
        tokenized_dataset: The tokenized dataset to be split.
    Returns:
        train_dataset: The training set.
        eval_dataset: The evaluation set.
    """
    # Split the original training set
    split_datasets = tokenized_dataset.train_test_split(test_size=640, seed=42)
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']
    
    return train_dataset, eval_dataset

def setup_training_args(training_args=None, **kwargs):
    """
    Setup training arguments for the Trainer.
    
    Args:
        training_args: Optional pre-configured TrainingArguments object.
        **kwargs: Keyword arguments to pass to TrainingArguments if training_args is None.
                  These will override default values.
                  
    Returns:
        training_args: Configured TrainingArguments for the Trainer.
    """
    if training_args is not None:
        return training_args
    
    # Default values
    default_args = {
        "output_dir": "output",
        "evaluation_strategy": "steps",
        "logging_steps": 100,
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "max_steps": 1000,
        "dataloader_num_workers": 4,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "optim": "adamw_torch",
        "gradient_checkpointing": False,
        "gradient_checkpointing_kwargs": {'use_reentrant': True},
        "report_to": "none"
    }
    
    # Override defaults with provided kwargs
    default_args.update(kwargs)
    
    # Initialize TrainingArguments with combined arguments
    training_args = TrainingArguments(**default_args)
    
    return training_args

def compute_metrics(pred):
    """
    Compute metrics for the evaluation.
    Args:
        pred: Predictions from the model containing labels and predictions.
    Returns:
        A dictionary containing the computed accuracy.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy
    }


def setup_trainer(model, train_dataset, eval_dataset, data_collator, training_args, compute_metrics, id2label=None):
    """
    Setup the Trainer for training and evaluation.
    Args:
        model: The model to be trained.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        data_collator: Data collator for batching the data.
        training_args: Training arguments for the Trainer.
        compute_metrics: Function to compute metrics during evaluation.
        id2label: Optional mapping from label IDs to class names.
    Returns:
        trainer: Configured Trainer instance.
    """
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    
    # Add label_names if available to avoid PEFT warning
    if id2label:
        label_names = [id2label[i] for i in sorted(id2label.keys())]
        trainer_kwargs["label_names"] = label_names
    
    # Initialize the Trainer
    trainer = Trainer(**trainer_kwargs)
    
    return trainer

def get_datasets():
    # Load and preprocess the dataset
    tokenized_dataset, data_collator = load_data()
    
    # Split the dataset into training and evaluation sets
    train_dataset, eval_dataset = split_dataset(tokenized_dataset)
    
    return train_dataset, eval_dataset, data_collator

def train(model, train_dataset, eval_dataset, data_collator, training_args, id2label=None):
    """
    Train the model using the Trainer.
    Args:
        model: The model to be trained.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        data_collator: Data collator for batching the data.
        training_args: Training arguments for the Trainer.
        compute_metrics: Function to compute metrics during evaluation.
        id2label: Optional mapping from label IDs to class names.
    """
    # Setup the Trainer with id2label
    trainer = setup_trainer(model, train_dataset, eval_dataset, data_collator, 
                          training_args, compute_metrics, id2label)
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    # Example LoRA configuration
    from peft import LoraConfig
    loraConfig = LoraConfig(
        r=8,  # Rank of the LoRA layers
        lora_alpha=16,  # Scaling factor for LoRA
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        bias="none",  # Bias handling in LoRA layers
        task_type="SEQ_CLS",  # Task type for sequence classification
        target_modules=['query']
    )
    # Setup training arguments
    training_args = setup_training_args()
    
    # Load and setup the LoRA model - note we're getting both model and id2label now
    from src.model import setup_lora_model
    model, id2label = setup_lora_model(loraConfig)

    # Load datasets
    train_dataset, eval_dataset, data_collator = get_datasets()

    # Train the model with id2label
    train(model, train_dataset, eval_dataset, data_collator, training_args, id2label)


