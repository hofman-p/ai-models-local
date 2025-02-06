from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model from local directory
model_dir = "./quantized_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Initialize pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Define the prompt
prompt = "You are a math-solving assistant. Only answer questions related to mathematics. If the question is not about math, say: 'I'm sorry, I only answer math-related questions. User: What is the capital of France ?"

# Generate the response
sequences = pipe(
    prompt,
    do_sample=True,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
)

# Output the generated text
print("Generated Text:", sequences[0]['generated_text'])
