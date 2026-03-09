import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    HF_TOKEN,
    MERGED_MODEL_DIR,
    OLLAMA_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

def load_model():
    """Load the merged model and tokenizer."""
    logger.info(f"Loading model from {MERGED_MODEL_DIR}")

    model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_MODEL_DIR),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MERGED_MODEL_DIR),
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def test_inference():
    """Test the model's inference capabilities."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting inference test...")

    # Load model
    model, tokenizer = load_model()

    # Test prompts
    test_prompts = [
        "Solve the Navier-Stokes equation for incompressible flow.",
        "Calculate the Schwarzschild radius for a 10 solar mass black hole.",
        "Derive the relativistic rocket equation.",
        "What is the Chandrasekhar limit?",
        "Explain Maxwell's equations.",
    ]

    # Generate responses
    for prompt in test_prompts:
        logger.info(f"\nPrompt: {prompt}")
        full_prompt = f"{OLLAMA_SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        response = generate_response(model, tokenizer, full_prompt)
        # Extract only the assistant's response
        assistant_response = response.split("Assistant:")[-1].strip()
        logger.info(f"Response: {assistant_response[:200]}...")

    logger.info("Inference test completed!")

if __name__ == "__main__":
    test_inference()