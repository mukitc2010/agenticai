import logging
from pathlib import Path

from config import (
    MERGED_MODEL_DIR,
    OLLAMA_MODEL_NAME,
    OLLAMA_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

def create_modelfile():
    """Create Ollama Modelfile for the fine-tuned model."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Creating Ollama Modelfile...")

    # Check if merged model exists
    if not MERGED_MODEL_DIR.exists():
        raise FileNotFoundError(f"Merged model not found at {MERGED_MODEL_DIR}")

    # Create Modelfile content
    modelfile_content = f"""FROM {MERGED_MODEL_DIR}

SYSTEM \"\"\"{OLLAMA_SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
PARAMETER repeat_last_n 64
"""

    # Write Modelfile
    modelfile_path = Path("Modelfile")
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    logger.info(f"Modelfile created at {modelfile_path}")
    logger.info("To create the Ollama model, run:")
    logger.info(f"ollama create {OLLAMA_MODEL_NAME} -f Modelfile")
    logger.info(f"Then run: ollama run {OLLAMA_MODEL_NAME}")

if __name__ == "__main__":
    create_modelfile()