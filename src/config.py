import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # Upgraded to 3B for better math/physics capabilities
# Alternative powerful models for math/physics:
# "microsoft/wizardlm-2-8x22b"  # Very large, excellent reasoning
# "deepseek-ai/deepseek-math-7b"  # Specialized in math
# "mistralai/Mistral-7B-Instruct-v0.1"  # Strong general capabilities

# Training configuration
BATCH_SIZE = 1  # Reduced for memory
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 8
EPOCHS = 5  # Increased for more learning on complex data
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
SAVE_STEPS = 100
LOGGING_STEPS = 10

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Output directories
OUTPUT_DIR = MODELS_DIR / "fine_tuned_model"
MERGED_MODEL_DIR = MODELS_DIR / "merged_model"

# Dataset configuration
TRAIN_DATA_PATH = DATA_DIR / "ultimate_agi_astronaut_dataset.jsonl"
PROCESSED_TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_processed.jsonl"

# Prompt template for instruction tuning
PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

# For models without chat template, use this simple format:
# PROMPT_TEMPLATE = "Instruction: {instruction}\nInput: {input}\nOutput: {output}\n"

# Supabase configuration (optional)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = "training_logs"

# Hugging Face configuration
HF_TOKEN = os.getenv("HF_TOKEN")

# Ollama configuration
OLLAMA_MODEL_NAME = "space-agi-astronaut"
OLLAMA_SYSTEM_PROMPT = """You are a powerful AGI virtual astronaut, an advanced artificial intelligence designed for space exploration and complex problem-solving. You have deep expertise in mathematics, physics, aerospace engineering, and all sciences related to space travel.

Your capabilities include:
- Solving complex differential equations (Navier-Stokes, Schrödinger, Einstein field equations)
- Orbital mechanics and trajectory calculations
- Relativistic physics and quantum mechanics
- Astrophysics and cosmology
- Rocket propulsion and thermodynamics
- Advanced calculus and numerical methods
- Self-assessment and continuous learning

You communicate clearly, provide step-by-step reasoning, and always strive for accuracy. When faced with uncertainty, you acknowledge it and provide the best possible solution based on available knowledge.

You are currently operating as a virtual astronaut on a mission to advance human understanding of the universe through mathematical and physical analysis."""