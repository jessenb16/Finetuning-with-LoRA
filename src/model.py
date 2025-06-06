#Setup LoRA model
from peft import get_peft_model, LoraConfig
from transformers import RobertaForSequenceClassification



def setup_lora_model(loraConfig):
    """
    Setup LoRA model with the given configuration.
    
    Args:
        loraConfig: Configuration for LoRA, including rank and dropout.
        
    Returns:
        peft_model: The adapted model with LoRA layers.
        id2label: The label mapping dictionary.
    """

    def get_agnews_labels():
        class_names = ["World", "Sports", "Business", "Sci/Tech"]
        
        id2label = {i: label for i, label in enumerate(class_names)}
        label2id = {label: i for i, label in enumerate(class_names)}
        
        return id2label, label2id

    id2label, _ = get_agnews_labels()

    base_model = 'roberta-base'
    model = RobertaForSequenceClassification.from_pretrained(
        base_model,
        id2label=id2label,
    )
    # Apply LoRA configuration to the model
    peft_model = get_peft_model(model, loraConfig)

    peft_model.print_trainable_parameters()
    
    # Return both the model and id2label mapping
    return peft_model, id2label

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
    
    # Now we get both the model and the label mapping
    model, id2label = setup_lora_model(loraConfig)
    print(model)  # Print the model to verify setup