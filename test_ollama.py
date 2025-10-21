import ollama
response = ollama.generate(
    model="llama3.2",
    prompt="Bonjour")
print(response["response"])