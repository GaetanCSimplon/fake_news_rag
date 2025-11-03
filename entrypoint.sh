#!/bin/bash
# Script d'entrée pour démarrer Ollama et Streamlit

# Lancer le serveur Ollama en arrière-plan
echo "Démarrage du serveur Ollama en arrière-plan..."
/usr/local/bin/ollama serve &

# Attendre que le serveur Ollama démarre
echo "Attente de 5 secondes pour le démarrage d'Ollama..."
sleep 5

# Lancer Streamlit en tant que processus principal (PID 1):
echo "Démarrage de l'application Streamlit sur 0.0.0.0:8501..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0