import requests
import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent / "../data/processed/cleaned_df_true_chunks.csv"
file_path = file_path.resolve()

chunk_true = pd.read_csv(file_path)
print(chunk_true.head())

texts = chunk_true["text"].tolist()

def get_embedding_from_ollama(text, model="all-minilm"):
    url = "http://localhost:11434/embed"  
    payload = {"model": model, "input": text}
    response = requests.post(url, json=payload)
    response.raise_for_status()  
    return response.json()["embedding"]

embeddings = [get_embedding_from_ollama(text) for text in texts]

chunk_true["embedding"] = embeddings
print(chunk_true.head())
#chunk_true.to_csv("data/with_embeddings.csv", index=False)
