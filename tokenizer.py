from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Step 1: Load GPT-2 and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 does not have a pad token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 2: Load custom text dataset
dataset = load_dataset("text", data_files={"train": "chat_data.txt"})

# Step 3: Tokenize data
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-chatbot",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_dir="./logs",
)

# Step 5: Setup trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 6: Train the model
trainer.train()

# Step 7: Save the model
model.save_pretrained("custom-chatbot")
tokenizer.save_pretrained("custom-chatbot")

print("✅ Training complete! Model saved in 'custom-chatbot' folder.")
