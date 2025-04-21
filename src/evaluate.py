import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import pickle
import pandas as pd
from datasets import load_dataset, Dataset
import os

def evaluate_model(inference_model, dataset, labelled=True, batch_size=8, data_collator=None):
    """
    Evaluate a PEFT model on a dataset.

    Args:
        inference_model: The model to evaluate.
        dataset: The dataset (Hugging Face Dataset) to run inference on.
        labelled (bool): If True, the dataset includes labels and metrics will be computed.
                         If False, only predictions will be returned.
        batch_size (int): Batch size for inference.
        data_collator: Function to collate batches. If None, the default collate_fn is used.

    Returns:
        If labelled is True, returns a tuple (metrics, predictions)
        If labelled is False, returns the predictions.
    """
    # Create the DataLoader
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_model.to(device)
    inference_model.eval()

    all_predictions = []
    if labelled:
        metric = evaluate.load('accuracy')

    # Loop over the DataLoader
    for batch in tqdm(eval_dataloader):
        # Move each tensor in the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = inference_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.append(predictions.cpu())

        if labelled:
            # Expecting that labels are provided under the "labels" key.
            references = batch["labels"]
            metric.add_batch(
                predictions=predictions.cpu().numpy(),
                references=references.cpu().numpy()
            )

    # Concatenate predictions from all batches
    all_predictions = torch.cat(all_predictions, dim=0)

    if labelled:
        eval_metric = metric.compute()
        print("Evaluation Metric:", eval_metric)
        return eval_metric, all_predictions
    else:
        return all_predictions
    
def get_unlabelled_dataset(data_path, tokenizer=None, preprocess_fn=None, file_type="pickle"):
    """
    Load and preprocess unlabelled data for inference.
    
    Args:
        data_path (str): Path to the unlabelled data file
        tokenizer: Optional tokenizer for preprocessing
        preprocess_fn: Function to preprocess the data. If None and tokenizer is provided,
                       a default preprocessing function will be used
        file_type (str): Type of file to load ('pickle', 'csv', 'json', 'huggingface')
    
    Returns:
        dataset: Preprocessed dataset ready for inference
    """

    
    # Load dataset based on file type
    if file_type == "pickle":
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
    elif file_type == "csv":
        df = pd.read_csv(data_path)
        dataset = Dataset.from_pandas(df)
    elif file_type == "json":
        df = pd.read_json(data_path)
        dataset = Dataset.from_pandas(df)
    elif file_type == "huggingface":
        # Assumes data_path is in format "dataset_name" or "dataset_name:split"
        if ":" in data_path:
            dataset_name, split = data_path.split(":", 1)
            dataset = load_dataset(dataset_name, split=split)
        else:
            dataset = load_dataset(data_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Check if text column exists before preprocessing
    has_text = "text" in dataset.column_names
    if not has_text:
        print(f"Warning: 'text' column not found in dataset. Available columns: {dataset.column_names}")
    
    # Apply preprocessing if provided
    if preprocess_fn:
        if has_text:
            dataset = dataset.map(preprocess_fn, batched=True, remove_columns=["text"])
        else:
            dataset = dataset.map(preprocess_fn, batched=True)
    
    # Use default preprocessing if tokenizer is provided but no custom preprocessing
    elif tokenizer and has_text:
        def default_preprocess(examples):
            # Make sure we get input_ids, attention_mask, etc.
            return tokenizer(examples["text"], truncation=True, padding=True, return_tensors=None)
        
        dataset = dataset.map(default_preprocess, batched=True, remove_columns=["text"])
    
    # Verify that the dataset now has input_ids
    print(f"Processed dataset features: {dataset.column_names}")
    if "input_ids" not in dataset.column_names:
        raise ValueError("Processing did not generate 'input_ids'. Check tokenization process.")
    
    return dataset

def run_inference(
    model, 
    dataset, 
    output_path, 
    data_collator=None, 
    batch_size=8, 
    id_column=None
):
    """
    Run inference on unlabelled dataset and save predictions to CSV.
    
    Args:
        model: The model to use for inference
        dataset: The preprocessed dataset for inference
        output_path: Path to save the predictions CSV
        data_collator: Function to collate data batches
        batch_size: Batch size for inference
        id_column: Optional column name to use as ID in output
                  If None, sequential IDs will be generated
                  
    Returns:
        predictions: Model predictions
        output_file: Path to saved predictions file
    """
    # Run inference
    predictions = evaluate_model(model, dataset, labelled=False, 
                                batch_size=batch_size, data_collator=data_collator)
    
    # Prepare output dataframe
    if id_column and id_column in dataset.column_names:
        ids = dataset[id_column]
    else:
        ids = list(range(len(predictions)))
    
    # Convert predictions to standard format
    pred_values = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
    
    # Create dataframe and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_output = pd.DataFrame({
        'ID': ids,
        'Label': pred_values
    })
    df_output.to_csv(output_path, index=False)
    
    print(f"Inference complete. Predictions saved to {output_path}")
    return predictions, output_path

if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/model"  # Path to your trained model
    data_path = "path/to/unlabelled/data.csv"  # Path to unlabelled data
    output_path = "output/predictions.csv"  # Path to save predictions

    # Load the model
    model = torch.load(model_path)  # Adjust loading method as needed

    # Load and preprocess the dataset
    dataset = get_unlabelled_dataset(data_path, file_type="csv")

    # Run inference and save predictions
    run_inference(model, dataset, output_path, batch_size=16)
