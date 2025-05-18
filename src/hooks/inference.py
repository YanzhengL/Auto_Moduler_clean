import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors.torch import load_file

MODEL_PATH = os.path.abspath(".\models\model.safetensors")
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
# Load safetensors weights
state_dict = load_file(MODEL_PATH)
model.load_state_dict(state_dict, strict=False)

# Move to device
model.to('cpu')

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_text("The future of AI is"))
