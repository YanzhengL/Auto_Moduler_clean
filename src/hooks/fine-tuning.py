import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from src.model import load_deepseek_model
from datasets import DatasetDict




MODEL_PATH = "C:/Users/psxyl37/PycharmProjects/Demo_DSR1-Distill-1.5B/models/model.safetensors"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

# Load model and tokenizer
model, tokenizer = load_deepseek_model(model_path="../models/model.safetensors")

# Load dataset
dataset = load_dataset("text", data_files={"train": "C:/Users/psxyl37/PycharmProjects/Demo_DSR1-Distill-1.5B/data/digital_twin_paper.txt"})
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Split dataset (80% train, 20% eval)
split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.2)
# Extract train and eval datasets
train_data = split_dataset["train"]
eval_data = split_dataset["test"]

# Apply LoRA (Low-Rank Adaptation)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",  # Set to 'steps' and adjust evaluation steps if needed
    save_strategy="steps",  # Same as above
    save_steps=2000,  # Adjust depending on how often you want to save
    logging_steps=2000,  # Adjust to log more frequently or less frequently
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=2,
    learning_rate=1e-2,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    eval_dataset=eval_data,
)
start_training = time.time()
# Start training
trainer.train()

end_training = time.time()
# Calculate and print the elapsed time
training_time = end_training - start_training
print(f"Time taken to run train the model with 1 paper of data: {training_time:.4f} seconds")

# Save fine-tuned model
trainer.save_model("C:/Users/psxyl37/PycharmProjects/Demo_DSR1-Distill-1.5B/models/fine_tuned_deepseek_r1")

saved_time = time.time()
save_time = saved_time - end_training
print(f"Time taken to run train the model with 1 paper of data: {save_time:.4f} seconds")