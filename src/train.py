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
        """
        Preprocess the dataset by tokenizing the text and preparing labels.
        Args:
            examples: A batch of examples from the dataset.
        Returns:
            tokenized: Tokenized examples with labels.
        """
        tokenized = tokenizer(examples['text'], truncation=True, padding=True)
        return tokenized
    
    tokenized_dataset = dataset.map(preprocess, batched=True,  remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

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

def setup_training_args(
        output_dir = "output",
        eval_strategy = "steps",
        logging_steps = 100,
        learning_rate = 2e-5,
        num_train_epochs = 3,
        max_steps = 1000,
        dataload_num_workers = 4,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        optimizer = "adamw_torch",
        gradient_checkpoiting = False,
        gradient_checkpointing_kwags = {'use_reetrant': True}
):
    """
    Setup training arguments for the Trainer.
    Args:
        output_dir: Directory to save the model and logs.
        eval_strategy: Evaluation strategy during training.
        logging_steps: Frequency of logging during training.
        learning_rate: Learning rate for the optimizer.
        num_train_epochs: Number of epochs for training.
        max_steps: Maximum number of training steps.
        dataload_num_workers: Number of workers for data loading.
        per_device_train_batch_size: Batch size for training.
        per_device_eval_batch_size: Batch size for evaluation.
        optimizer: Optimizer type to use.
        gradient_checkpoiting: Whether to use gradient checkpointing.
        gradient_checkpointing_kwags: Additional arguments for gradient checkpointing.
    Returns:
        training_args: Configured TrainingArguments for the Trainer.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=eval_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        dataloader_num_workers=dataload_num_workers,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        optim=optimizer,
        gradient_checkpointing=gradient_checkpoiting,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwags
    )
    
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


def setup_trainer(model, train_dataset, eval_dataset, data_collator, training_args, compute_metrics):
    """
    Setup the Trainer for training and evaluation.
    Args:
        model: The model to be trained.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        data_collator: Data collator for batching the data.
        training_args: Training arguments for the Trainer.
        id2label: Mapping from label IDs to class names.
        compute_metrics: Function to compute metrics during evaluation.
    Returns:
        trainer: Configured Trainer instance.
    """
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    return trainer

def get_datasets():
    # Load and preprocess the dataset
    tokenized_dataset, data_collator = load_data()
    
    # Split the dataset into training and evaluation sets
    train_dataset, eval_dataset = split_dataset(tokenized_dataset)
    
    return train_dataset, eval_dataset, data_collator

def train(model, train_dataset, eval_dataset, data_collator):
    # Setup the Trainer
    trainer = setup_trainer(model, train_dataset, eval_dataset, data_collator, training_args, compute_metrics)
    
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
    
    # Load and setup the LoRA model
    from src.model import setup_lora_model
    model = setup_lora_model(loraConfig)

    # Load datasets
    train_dataset, eval_dataset, data_collator = get_datasets()

    # Train the model
    train(model, train_dataset, eval_dataset, data_collator)


