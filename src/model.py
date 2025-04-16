#Setup LoRA model
from peft import get_peft_model, LoraConfig
from transformers import RobertaForSequenceClassification

def setup_lora_model(loraConfig, id2label):
    """
    Setup LoRA model with the given configuration.
    
    Args:
        model: The base model to be adapted with LoRA.
        loraConfig: Configuration for LoRA, including rank and dropout.
        
    Returns:
        The adapted model with LoRA layers.
    """
    base_model = 'roberta-base'
    model = RobertaForSequenceClassification.from_pretrained(
        base_model,
        id2label=id2label,
    )
    # Apply LoRA configuration to the model
    peft_model = get_peft_model(model, loraConfig)

    peft_model.print_trainable_parameters()
    
    return peft_model

if __name__ == "__main__":
    # Example LoRA configuration
    loraConfig = LoraConfig(
        r=8,  # Rank of the LoRA layers
        lora_alpha=16,  # Scaling factor for LoRA
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        bias="none",  # Bias handling in LoRA layers
        task_type="SEQ_CLS",  # Task type for sequence classification
        target_modules=['query']
    )

    # Example id2label mapping
    labels = ['World', 'Sports', 'Business', 'Sci/Tech']
    id2label = {i: label for i, label in enumerate(labels)}
    
    model = setup_lora_model(loraConfig, id2label)
    print(model)  # Print the model to verify setup