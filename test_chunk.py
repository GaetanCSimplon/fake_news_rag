from src.preprocessing import TextChunker
import pandas as pd

# Initialisation du chunker
chunker = TextChunker(chunk_size=300, overlap=50)

# Charger le DataFrame nettoyé
df = pd.read_csv("data/processed/cleaned_df_true.csv")

# Utiliser le texte de la première ligne pour tester
text = df.loc[0, "text"]

# Génération des chunks
chunks = chunker.split_text(text)

# Affichage des résultats
print(f"{len(chunks)} chunks générés.")
print("Longueur (en mots) de chaque chunk :", [len(c.split()) for c in chunks])

# (optionnel) Afficher un extrait pour vérifier
for i, c in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(c[:300], "...")  # n'afficher que les 300 premiers caractères
