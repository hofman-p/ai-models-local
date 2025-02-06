from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Charger le dataset spécialisé (Maths)
dataset = load_dataset("math_qa", split="train")

# Tokenisation
def tokenize_function(examples):
    return tokenizer(examples["question"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function)

training_args = TrainingArguments(
    output_dir="./maths-specialized-model",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
