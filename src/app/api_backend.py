from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the DeepSeek model and tokenizer
model_name = "deepseek-model"  # Update this with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Function to process user request and search local knowledge base
def process_user_request(user_input, knowledge_base):
    # Tokenize the input
    inputs = tokenizer.encode(user_input, return_tensors="pt")

    # Generate model output
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100)

    # Decode the model's output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Search knowledge base for relevant information (example: find the closest match)
    relevant_info = search_knowledge_base(response, knowledge_base)

    return response, relevant_info


# Function to search through a local knowledge base (simplified)
def search_knowledge_base(query, knowledge_base):
    # Example: match query to knowledge base entries
    matched_entries = [entry for entry in knowledge_base if query.lower() in entry.lower()]
    return matched_entries


# Example local knowledge base
knowledge_base = [
    "Power electronics are crucial in modern energy systems.",
    "DC-DC converters are used to step up or step down voltage in power systems.",
    "Anomaly detection in power systems can help predict failures."
]

# Example user input
user_input = "How do DC-DC converters work?"

# Process the user request
response, relevant_info = process_user_request(user_input, knowledge_base)

# Print results
print("Model Response:", response)
print("Relevant Knowledge Base Information:", relevant_info)
