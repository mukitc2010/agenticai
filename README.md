# LLM Fine-tuning Project: AGI Astronaut Model

This project implements a complete local LLM fine-tuning pipeline to create a powerful AGI space-based virtual astronaut model capable of solving complex mathematical physics problems.

## Features

- **LoRA Fine-tuning**: Efficient parameter adaptation using PEFT
- **Advanced Dataset**: 2000+ examples covering calculus, orbital mechanics, rocket equations, quantum physics, relativity, and more
- **Ollama Integration**: Deploy the model locally with custom system prompts
- **Self-Evaluation**: P95/P99 performance metrics for AGI readiness assessment
- **Supabase Logging**: Optional cloud logging for training metrics

## Model

- Base Model: Qwen2.5-3B-Instruct (upgraded for better math/physics capabilities)
- Fine-tuned for: Complex problem-solving in aerospace, physics, and mathematics
- System Prompt: Virtual astronaut role with space exploration expertise

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended) or MPS (Mac)
- 16GB+ RAM
- Ollama installed for local inference

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mukitc2010/agenticai.git
   cd agenticai
   ```

2. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your Hugging Face token and Supabase credentials
   ```

## Web Interface

A beautiful web interface is available for interacting with the AGI Astronaut:

### Live Demo (No Installation Required!)
Visit the live demo: [AGI Astronaut Web Interface](https://mukitc2010.github.io/agenticai/)

**The web interface now works in two modes:**

1. **Demo Mode** (No local setup needed)
   - Try example responses about physics and space topics
   - No Ollama installation required
   - Perfect for exploring the interface

2. **Full Mode** (With Local Ollama)
   - Unlimited AI responses
   - Connect your local Ollama instance
   - Full AGI Astronaut intelligence

### Features
✅ Works without Ollama (demo mode with examples)  
✅ Auto-detects local Ollama when running  
✅ Beautiful space-themed interface  
✅ Responsive design for all devices  
✅ Real-time connection status  
✅ Zero installation for demo  

### Using the Web Interface

**Option 1: Demo Mode (Easiest)**
- Visit [https://mukitc2010.github.io/agenticai/](https://mukitc2010.github.io/agenticai/)
- Try asking about Schwarzschild radius, orbits, quantum mechanics, etc.
- See example AI responses

**Option 2: Full Mode with Local Ollama**
1. Install [Ollama](https://ollama.ai)
2. Run the model: `ollama run space-agi-astronaut`
3. Visit the web interface - it will auto-detect the connection
4. Enjoy unlimited AGI Astronaut intelligence!

### Local Web Interface
1. Clone the repo: `git clone https://github.com/mukitc2010/agenticai.git`
2. Open `docs/index.html` in your browser
3. The interface will automatically detect local Ollama if running



## Ollama Deployment

After training, create the model in Ollama:
```bash
ollama create space-agi-astronaut -f Modelfile
ollama run space-agi-astronaut
```

## Usage

### Training Pipeline

Run the complete pipeline:
```bash
bash scripts/run_all.sh
```

This will:
- Prepare the dataset
- Fine-tune the model with LoRA
- Merge adapters
- Generate Ollama Modelfile
- Test inference

### Individual Steps

1. Prepare dataset:
   ```bash
   python src/dataset_utils.py
   ```

2. Train model:
   ```bash
   python src/train_lora.py
   ```

3. Merge and export:
   ```bash
   python src/export_model.py
   ```

4. Test model:
   ```bash
   python src/test_inference.py
   ```

5. Evaluate AGI capabilities:
   ```bash
   python src/evaluate_agi.py
   ```

### Web Interface

Open `docs/index.html` in your browser to chat with the AGI Astronaut.

## Dataset

The dataset includes advanced problems in:
- Navier-Stokes equations
- Schwarzschild radius calculations
- Relativistic rocket equations
- Three-body problem solutions
- Chandrasekhar limits
- Maxwell's equations
- Hawking temperature
- Virial theorem
- Heat equation in astrophysics
- Roche limits
- Schrödinger equation
- Cosmological constant
- Tsiolkovsky rocket equation
- Einstein field equations
- Jeans length
- Boltzmann equation
- Chandrasekhar mass
- Stefan-Boltzmann law
- Poisson equation
- Nuclear fusion lifetimes
- Dirac equation
- Olbers' paradox
- Bernoulli equation
- Kepler problem in GR
- Planck length
- Fokker-Planck equation
- Eddington luminosity
- Saha equation
- Lane-Emden equation
- Hubble constant
- Yang-Mills equations
- Debye length
- Friedmann equation
- Hartree-Fock method
- Tolman-Oppenheimer-Volkoff limit
- Boltzmann transport
- Schwarzschild-de Sitter metric
- Klein-Gordon equation
- Jeans instability
- Boundary layer theory
- And many more...

## Configuration

Edit `src/config.py` to modify:
- Model parameters
- Training hyperparameters
- Dataset paths
- LoRA configuration

## Evaluation

The model is evaluated on:
- Mathematical accuracy
- Physical reasoning
- Problem complexity handling
- AGI readiness metrics (P95/P99)

## Architecture

```
data/
├── agi_astronaut_dataset.jsonl      # Original dataset
├── ultimate_agi_astronaut_dataset.jsonl  # Expanded dataset
└── processed/
    └── train_processed.jsonl        # Tokenized data

src/
├── config.py                        # Configuration
├── dataset_utils.py                 # Data preparation
├── train_lora.py                    # LoRA training
├── export_model.py                  # Model merging
├── test_inference.py                # Inference testing
├── evaluate_agi.py                  # AGI evaluation
├── ollama_prep.py                   # Ollama setup
├── supabase_utils.py                # Logging utilities
└── __init__.py

models/
├── fine_tuned_model/                # LoRA adapters
├── merged_model/                    # Full model
└── quantized_model/                 # GGUF for Ollama

scripts/
└── run_all.sh                       # Complete pipeline
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Based on Qwen2.5 models from Alibaba Cloud
- Uses PEFT, TRL, and Transformers libraries
- Inspired by advances in AGI and space exploration