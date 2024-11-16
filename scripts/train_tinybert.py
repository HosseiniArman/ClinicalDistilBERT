import transformers as ts
from tinybert_distillation import TinyBERTDistillation
import torch
from datasets import load_from_disk

def main():
    # Paths
    save_path = "ClinicalModels/models/ClinicalTinyBERT/"  # Directory to save checkpoints and final model
    teacher_path = "emilyalsentzer/Bio_ClinicalBERT"       # Pretrained teacher model
    dataset_path = "tokenizedDatasets/mimic-large/"        # Preprocessed dataset path

    # Load dataset and tokenizer
    print("Loading dataset...")
    ds = load_from_disk(dataset_path)
    tokenizer = ts.AutoTokenizer.from_pretrained(teacher_path)
    print("Dataset and tokenizer loaded successfully.")

    # Load teacher model
    print("Initializing teacher model...")
    teacher = ts.AutoModelForMaskedLM.from_pretrained(teacher_path)
    for param in teacher.parameters():
        param.requires_grad = False  # Freeze teacher parameters during training

    # Initialize student model
    print("Initializing student model...")
    pretrained_config = ts.AutoConfig.from_pretrained("albert-base-v1")
    pretrained_config.num_hidden_layers = 6  # Reduced layers for TinyBERT
    pretrained_config.hidden_size = teacher.config.hidden_size
    student = ts.AutoModelForMaskedLM.from_config(pretrained_config)

    # Initialize the distillation model
    print("Initializing distillation wrapper...")
    model = TinyBERTDistillation(student=student, teacher=teacher)

    # Training arguments
    print("Setting up training arguments...")
    training_args = ts.TrainingArguments(
        output_dir=save_path + "checkpoints",
        num_train_epochs=3,                        # Number of epochs
        learning_rate=5e-4,                        # Learning rate
        per_device_train_batch_size=16,            # Batch size per device
        save_steps=1000,                           # Steps to save checkpoints
        save_total_limit=2,                        # Limit number of saved checkpoints
        logging_steps=100,                         # Log progress every 100 steps
        evaluation_strategy="no",                 # No evaluation during training
        overwrite_output_dir=True,                # Overwrite old output directory
    )

    # Data collator for masked language modeling
    print("Preparing data collator...")
    data_collator = ts.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
    )

    # Trainer for distillation
    print("Initializing trainer...")
    trainer = ts.Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    # Training process
    print("Starting training...")
    trainer.train()

    # Save the final model
    print(f"Saving final model to {save_path + 'final_model/' }...")
    trainer.save_model(save_path + "final_model/")
    print("Training and saving complete.")

if __name__ == "__main__":
    main()
