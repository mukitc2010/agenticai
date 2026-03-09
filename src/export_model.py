import logging
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    HF_TOKEN,
    MERGED_MODEL_DIR,
    MODEL_NAME,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

def merge_lora_adapters():
    """Merge LoRA adapters with base model."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting LoRA adapter merging...")

    # Load base model
    logger.info(f"Loading base model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
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

    # Load LoRA model
    logger.info(f"Loading LoRA adapters from {OUTPUT_DIR}")
    model = PeftModel.from_pretrained(model, str(OUTPUT_DIR))

    # Merge adapters
    logger.info("Merging adapters...")
    merged_model = model.merge_and_unload()

    # Save merged model
    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving merged model to {MERGED_MODEL_DIR}")
    merged_model.save_pretrained(str(MERGED_MODEL_DIR))
    tokenizer.save_pretrained(str(MERGED_MODEL_DIR))

    logger.info("Model merging completed!")

if __name__ == "__main__":
    merge_lora_adapters()