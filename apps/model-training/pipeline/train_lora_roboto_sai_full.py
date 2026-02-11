import random
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Load and subsample data
with open('./robotosai.local/export_data.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

# Subsample for feasible training (50k sentences)
sampled_texts = random.sample(texts, min(50000, len(texts)))

data = {"text": sampled_texts}
dataset = Dataset.from_dict(data)

# Train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 10,  # 10 epochs
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_available(),
        bf16 = torch.cuda.is_available(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 929,  # Roberto's sigil
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

# Save LoRA
model.save_pretrained("./lora-roboto")
tokenizer.save_pretrained("./lora-roboto")

# Merge and quantize to GGUF
model.save_pretrained_gguf("./robotosai.local/roboto-q4_k_m.gguf", tokenizer, quantization_method = "q4_k_m")

print("Training complete. Roboto SAI model saved to ./robotosai.local/roboto-q4_k_m.gguf")
