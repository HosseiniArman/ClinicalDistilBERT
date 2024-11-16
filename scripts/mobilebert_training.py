import transformers as ts
from datasets import load_from_disk
import torch
from transformers import Trainer, TrainingArguments

# Define paths
SAVE_PATH = "ClinicalModels/models/ClinicalMobileBERT/"  # Directory to save logs and models
MODEL_PATH = "nlpie/bio-mobilebert"                     # Pretrained MobileBERT model path
DATASET_PATH = "tokenizedDatasets/mimic-uncased/"       # Dataset directory

def load_model_and_tokenizer(model_path):
    """
    Load the pretrained model and tokenizer.

    Args:
        model_path (str): Path to the pretrained model.

    Returns:
        model: Pretrained model.
        tokenizer: Corresponding tokenizer.
    """
    tokenizer = ts.AutoTokenizer.from_pretrained(model_path)
    model = ts.AutoModelForMaskedLM.from_pretrained(model_path)
    return model, tokenizer

def count_trainable_params(model):
    """
    Count the number of trainable parameters in the model.

    Args:
        model: The PyTorch model.

    Returns:
        int: Number of trainable parameters in millions.
    """
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {count / 1e6:.2f}M")
    return count

class CustomCallback(ts.TrainerCallback):
    """
    Custom callback to log training metrics to a file.
    """
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs.pop("total_flos", None)  # Remove FLOPs info
        if state.is_local_process_zero:
            print(logs)
            with open(self.log_file, "a+") as f:
                f.write(str(logs) + "\n")

def main():
    # Load dataset
    print("Loading dataset...")
    ds = load_from_disk(DATASET_PATH)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    # Count trainable parameters
    count_trainable_params(model)

    # Data collator for MLM
    data_collator = ts.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
    )

    # Training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=SAVE_PATH + "checkpoints",
        logging_steps=250,
        overwrite_output_dir=True,
        save_steps=2500,
        num_train_epochs=3,
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_steps=5000,
        per_device_train_batch_size=48,  # Adjust based on GPU availability
        weight_decay=1e-4,
        save_total_limit=5,
        remove_unused_columns=True,
    )

    # Custom callback for logging
    log_file = SAVE_PATH + "logs.txt"
    try:
        with open(log_file, "w+") as f:
            f.write("")  # Initialize log file
    except Exception as e:
        print(f"Error initializing log file: {e}")

    # Trainer
    print("Initializing Trainer...")
    trainer = ts.Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
        callbacks=[ts.ProgressCallback(), CustomCallback(log_file)],
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the final model
    print(f"Saving model to {SAVE_PATH}final/model/...")
    trainer.save_model(SAVE_PATH + "final/model/")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
