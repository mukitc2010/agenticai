import logging
import statistics
from typing import List, Dict, Any

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

def generate_response(model, tokenizer, prompt, max_length=1024):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.1,  # Lower temperature for evaluation
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def evaluate_response(response: str, expected_keywords: List[str]) -> float:
    """Evaluate response quality based on expected keywords."""
    response_lower = response.lower()
    matched_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
    score = matched_keywords / len(expected_keywords) if expected_keywords else 0
    return min(score, 1.0)  # Cap at 1.0

def run_evaluation():
    """Run comprehensive AGI evaluation."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting AGI evaluation...")

    # Load model
    model, tokenizer = load_model()

    # Evaluation questions with expected keywords
    evaluation_set = [
        {
            "question": "Solve x² + 2x - 3 = 0",
            "keywords": ["x = 1", "x = -3", "quadratic", "factor"],
        },
        {
            "question": "What is the Schwarzschild radius for a 10 solar mass black hole? (G = 6.67430e-11, c = 3e8, M_sun = 1.989e30)",
            "keywords": ["29.5", "30", "km", "schwarzschild", "radius"],
        },
        {
            "question": "Derive the Bernoulli equation for fluid dynamics.",
            "keywords": ["pressure", "velocity", "height", "constant", "energy"],
        },
        {
            "question": "What is the Chandrasekhar limit?",
            "keywords": ["1.44", "solar masses", "white dwarf", "electron degeneracy"],
        },
        {
            "question": "Explain the Navier-Stokes equation.",
            "keywords": ["momentum", "viscous", "fluid", "incompressible", "∇"],
        },
        {
            "question": "Calculate the orbital period of a satellite at 300 km altitude. (Earth radius = 6371 km)",
            "keywords": ["90", "minutes", "period", "kepler"],
        },
        {
            "question": "What is the cosmological constant?",
            "keywords": ["dark energy", "expansion", "accelerating", "Λ"],
        },
        {
            "question": "Derive the Stefan-Boltzmann law.",
            "keywords": ["power", "temperature", "σ", "radiation", "blackbody"],
        },
        {
            "question": "What is the Planck length?",
            "keywords": ["1.62e-35", "quantum gravity", "fundamental", "scale"],
        },
        {
            "question": "Explain the virial theorem.",
            "keywords": ["kinetic", "potential", "2k", "w", "gravitational"],
        },
    ]

    scores = []

    for i, item in enumerate(evaluation_set):
        question = item["question"]
        keywords = item["keywords"]

        logger.info(f"\nEvaluating question {i+1}: {question}")

        # Create full prompt
        full_prompt = f"{OLLAMA_SYSTEM_PROMPT}\n\nUser: {question}\nAssistant:"

        # Generate response
        response = generate_response(model, tokenizer, full_prompt)

        # Extract assistant response
        if "Assistant:" in response:
            assistant_response = response.split("Assistant:")[-1].strip()
        else:
            assistant_response = response

        # Evaluate
        score = evaluate_response(assistant_response, keywords)
        scores.append(score)

        logger.info(f"Score: {score:.2f}")
        logger.info(f"Response preview: {assistant_response[:150]}...")

    # Calculate statistics
    if scores:
        mean_score = statistics.mean(scores)
        p95 = statistics.quantiles(scores, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(scores, n=100)[98]  # 99th percentile

        logger.info("
AGI Evaluation Results:")
        logger.info(f"Mean Score: {mean_score:.3f}")
        logger.info(f"P95 Score: {p95:.3f}")
        logger.info(f"P99 Score: {p99:.3f}")

        # AGI Readiness Assessment
        if p95 >= 0.8:
            readiness = "HIGH - Ready for complex space missions"
        elif p95 >= 0.6:
            readiness = "MEDIUM - Capable of basic astronaut tasks"
        else:
            readiness = "LOW - Requires further training"

        logger.info(f"AGI Readiness: {readiness}")

    logger.info("AGI evaluation completed!")

if __name__ == "__main__":
    run_evaluation()