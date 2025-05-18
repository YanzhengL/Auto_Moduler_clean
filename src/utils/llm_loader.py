from safetensors import safe_open
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file


# Path to the downloaded model.safetensors file
#file_path = "../models/model.safetensors"
# Load the model weights
#with safe_open(file_path, framework="pt") as f:  # Use "pt" for PyTorch, "tf" for TensorFlow
#    for key in f.keys():
#        tensor = f.get_tensor(key)
#        print(f"Loaded tensor {key} with shape {tensor.shape}")


def load_deepseek_model(model_name="Qwen/Qwen2.5-Math-1.5B", model_path="../models/model.safetensors", device=None):
    """
    Loads the DeepSeek-R1 model from safetensors and returns the model and tokenizer.

    Args:
        model_name (str): The base model name from Hugging Face.
        model_path (str): Path to the `.safetensors` file.
        device (str): Device to load the model onto ('cuda' or 'cpu'). If None, it auto-detects.

    Returns:
        model (AutoModelForCausalLM): The loaded DeepSeek model.
        tokenizer (AutoTokenizer): Tokenizer for text processing.
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model architecture
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    # Load safetensors weights
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)

    # Move to device
    model.to(device)

    return model, tokenizer

# Load model and tokenizer
#model, tokenizer = load_deepseek_model(model_path="../models/model.safetensors")

# Generate text
#def generate_text(prompt, model, tokenizer, max_length=100, device="cpu"):
#    inputs = tokenizer(prompt, return_tensors="pt").to(device)
#    with torch.no_grad():
#        output = model.generate(**inputs, max_length=max_length)
#    return tokenizer.decode(output[0], skip_special_tokens=True)
#
#print(generate_text("Digital Twin is", model, tokenizer))