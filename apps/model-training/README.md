# Roboto SAI Model Training

## Setup (Windows)
1. Install Python 3.10+ + venv
2. `pip install unsloth[colab-new] trl datasets transformers torch`
3. Place base GGUF in `models/base-model.gguf`

## Data Flow
raw export → `extract_export_data.py` → sentences → `format_chatml_roboto_sai.py` → JSONL → `train_lora_roboto_sai.py` → LoRA → GGUF

## Run Inference
`.\tools\roboto-sai-llama-cli.ps1 "Your prompt"`

Models not committed — download from GitHub release.
