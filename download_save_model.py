from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

login(hugging_face_api_key)

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Download Mistral7B from Hugging Face
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Save the quantized model and tokenizer locally
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")