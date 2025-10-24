import chromadb
from chromadb.config import Settings

# إنشاء قاعدة بيانات محلية
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

collection = client.create_collection("my_texts")

# إضافة النصوص والمتجهات
collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"source": "csv_file"} for _ in texts],
    ids=[str(i) for i in range(len(texts))]
)
