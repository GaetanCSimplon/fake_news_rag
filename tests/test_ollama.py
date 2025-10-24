import ollama

models = ollama.list()["models"]
for model in models:
    print("-", model["model"])

response = ollama.generate(
    model="llama3.2",
    prompt="Samy est nul")
print(response["response"])

