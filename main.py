from transformers import LlamaForCausalLM, LlamaTokenizer

# Load the model and tokenizer
model_path = "path/to/your/downloaded/llama2/model"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model
response = generate_response("Hello, I'm working on a JARVIS-like AI. Can you help me?")
print(response)