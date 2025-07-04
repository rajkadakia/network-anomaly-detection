import os
from huggingface_hub import constants as hf_constants

# Method 1: Check environment variables
print("HF_HOME:", os.environ.get("HF_HOME", "~/.cache/huggingface"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE", "default location"))

# Method 2: Use huggingface_hub constants
print("HF Hub Cache:", hf_constants.HF_HUB_CACHE)