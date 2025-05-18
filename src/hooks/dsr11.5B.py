from huggingface_hub import hf_hub_download

# Replace with your model's details
repo_id = "Qwen/Qwen2.5-Math-1.5B"
#filename = "model.safetensors"
filenames = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]

# Download the file
for filename in filenames:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"File downloaded to: {file_path}")