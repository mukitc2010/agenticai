import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from config import (
    BATCH_SIZE,
    EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    HF_TOKEN,
    LEARNING_RATE,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    OUTPUT_DIR,
    PROCESSED_TRAIN_DATA_PATH,
    SAVE_STEPS,
    SUPABASE_KEY,
    SUPABASE_TABLE,
    SUPABASE_URL,
    TARGET_MODULES,
)

try:
    from supabase_utils import insert_rows
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase not available, logging disabled")

logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_model_and_tokenizer():
    """Load the model and tokenizer with quantization."""
    logger.info(f"Loading model: {MODEL_NAME}")

    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded. Device: {model.device}")
    return model, tokenizer

def setup_lora_config():
    """Configure LoRA parameters."""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    return lora_config

def prepare_model_for_training(model, lora_config):
    """Prepare model for LoRA training."""
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model

def load_training_data(tokenizer):
    """Load and tokenize the training dataset."""
    logger.info(f"Loading dataset from {PROCESSED_TRAIN_DATA_PATH}")

    dataset = load_dataset(
        "json",
        data_files=str(PROCESSED_TRAIN_DATA_PATH),
        split="train"
    )

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset

def setup_training_args():
    """Set up training arguments."""
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",  # Disable wandb/tensorboard
    )
    return training_args

def log_to_supabase(epoch, loss, step):
    """Log training metrics to Supabase."""
    if not SUPABASE_AVAILABLE or not SUPABASE_URL or not SUPABASE_KEY:
        return

    try:
        data = [{
            "epoch": epoch,
            "loss": loss,
            "step": step,
            "model": MODEL_NAME,
        }]
        insert_rows(SUPABASE_TABLE, data)
        logger.info(f"Logged to Supabase: epoch {epoch}, loss {loss:.4f}")
    except Exception as e:
        logger.error(f"Failed to log to Supabase: {e}")

def train_lora():
    """Main training function."""
    setup_logging()
    logger.info("Starting LoRA fine-tuning...")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Setup LoRA
    lora_config = setup_lora_config()
    model = prepare_model_for_training(model, lora_config)

    # Load data
    train_dataset = load_training_data(tokenizer)

    # Training arguments
    training_args = setup_training_args()

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the model
    trainer.save_model(str(OUTPUT_DIR))
    logger.info(f"Model saved to {OUTPUT_DIR}")

    # Log final metrics
    log_to_supabase(EPOCHS, trainer.state.log_history[-1].get("train_loss", 0), trainer.state.global_step)

    logger.info("Training completed!")

if __name__ == "__main__":
    train_lora()