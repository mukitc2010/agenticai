import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from config import (
    TRAIN_DATA_PATH,
    PROCESSED_TRAIN_DATA_PATH,
    PROMPT_TEMPLATE,
    MAX_SEQ_LENGTH
)

logger = logging.getLogger(__name__)

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} samples from {file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        raise
    return data

def preprocess_sample(sample: Dict[str, Any]) -> str:
    """Convert a sample to the prompt format."""
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')

    # Format using the prompt template
    formatted = PROMPT_TEMPLATE.format(
        instruction=instruction,
        input=input_text,
        output=output
    )
    return formatted

def save_processed_data(data: List[str], output_path: Path):
    """Save processed data to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump({"text": item}, f)
            f.write('\n')
    logger.info(f"Saved {len(data)} samples to {output_path}")

def main():
    """Main data preparation function."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting dataset preparation...")

    # Load raw data
    raw_data = load_jsonl(TRAIN_DATA_PATH)

    # Preprocess each sample
    processed_data = []
    for sample in raw_data:
        processed_text = preprocess_sample(sample)
        if len(processed_text) <= MAX_SEQ_LENGTH * 4:  # Rough character limit
            processed_data.append(processed_text)
        else:
            logger.warning(f"Skipping sample too long: {len(processed_text)} chars")

    # Save processed data
    save_processed_data(processed_data, PROCESSED_TRAIN_DATA_PATH)

    logger.info("Dataset preparation completed")

if __name__ == "__main__":
    main()