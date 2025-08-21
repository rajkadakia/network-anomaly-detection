import os
from huggingface_hub import constants as hf_constants
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Directory to save the model
model_dir = "../models/flan-t5-base"
os.makedirs(model_dir, exist_ok=True)

# Load and save tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
tokenizer.save_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model.save_pretrained(model_dir)
print(f"flan-t5-base downloaded to {model_dir}")

# Method 1: Check environment variables
print("HF_HOME:", os.environ.get("HF_HOME", "~/.cache/huggingface"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE", "default location"))

# Method 2: Use huggingface_hub constants
print("HF Hub Cache:", hf_constants.HF_HUB_CACHE)