import re
import chromadb
import ollama
from src.embedding import OllamaEmbedder

# client = chromadb.Client(Settings(persist_directory="data/vector_store"))

# Initialize Chroma client and collection
client = chromadb.PersistentClient(path="data/vector_db")
print(client.list_collections())
collection_name = "articles"
collection = client.get_collection(name=collection_name)

# --- Input ---
#user_input = "Breaking news: Scientists discovered a new bird species in the Amazon rainforest. It has colorful feathers and a rare song pattern."
user_input = ""
# --- Cleaning ---
def clean(text: str):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

cleaned_text = clean(user_input)

# --- Embedding ---
embedding_result = ollama.embed(
    model="all-minilm",
    input=cleaned_text
)

embedding_vector = embedding_result["embeddings"][0]  # extract list of floats

# --- Normalize ---
olla = OllamaEmbedder()
normalized_vector = olla.normalize_vector(embedding_vector)

# --- Query ---
query_results = collection.query(
    query_embeddings=[normalized_vector],
    n_results=1
)

# --- Response ---
if query_results['documents'] and query_results['documents'][0]:
    data = query_results['documents'][0][0]
    print(data)
else:
    data = "No relevant data found."

#prompt=f"Using this data: {data}. Respond to this prompt: {user_input}"
prompt = (
        "You are given a candidate article (User Article) and a context composed of similar "
        "document chunks retrieved from a trusted article database. Use the context to judge "
        "whether the User Article is TRUE (credible) or FAKE (misinformation). "
        "Return exactly one of the labels: TRUE or FAKE, then on the next lines provide a "
        "brief justification (2-4 bullet points) citing which context chunks support your decision.\n\n"
        "CONTEXT (retrieved chunks):\n"
        f"{data}\n\n"
        "USER ARTICLE:\n"
        f"{user_input}\n\n"
        "OUTPUT FORMAT:\n"
        "Label: <TRUE or FAKE>\n"
        "Justification:\n"
        "- bullet 1 (cite source header or chunk snippet)\n"
        "- bullet 2\n\n"
        "Be concise, factual, and do not hallucinate. If the evidence is inconclusive, say 'INCONCLUSIVE' "
        "instead of TRUE/FAKE and explain why.\n\n"
        "Start now.\n"
    )
print(f"this is the promt {prompt}")

output = ollama.generate(
    model="phi3:mini",
    prompt=prompt
)

#print(output)
print(output.get("response"))
