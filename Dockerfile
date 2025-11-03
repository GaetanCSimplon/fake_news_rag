# =======================================================
# Étape 1 : Préparation - Installer Ollama et les Modèles
# =======================================================
FROM ollama/ollama:latest AS ollama-builder

# Ajout d'une seule commande RUN pour démarrer le serveur, pull, et nettoyer.
RUN ollama serve & \
    # Attendre que le serveur démarre (facultatif mais plus sûr)
    sleep 5 && \
    # Tirer les modèles LLM.
    ollama pull all-minilm && \
    ollama pull llama3.2 && \
    # Arrêter proprement le serveur Ollama qui tourne en arrière-plan
    pkill ollama

# ... (Première partie inchangée, incluant l'étape ollama-builder)

# =======================================================
# Étape 2 : Construction de l'image finale (Python + Code)
# =======================================================
FROM python:3.12

WORKDIR /fake_news_rag
ENV OLLAMA_MODELS=/root/.ollama/models
ENV OLLAMA_HOST=0.0.0.0:11434

# --- Copie des dépendances Python ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copie du binaire Ollama depuis l'étape de build ---
# Utilisation de /usr/bin/ollama (le chemin probable dans l'image ollama/ollama)
COPY --from=ollama-builder /usr/bin/ollama /usr/local/bin/ollama 
# Copie les modèles pré-téléchargés (plusieurs Go)
COPY --from=ollama-builder /root/.ollama/ /root/.ollama/

# --- Copie du code et du script de démarrage ---
COPY . .
RUN chmod +x entrypoint.sh

# --- Configuration des Ports et du Volume ---
EXPOSE 8501
EXPOSE 11434
VOLUME /root/.ollama 

# --- Commande de Démarrage ---
CMD ["./entrypoint.sh"]