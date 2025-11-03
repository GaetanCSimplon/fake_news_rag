#!/bin/bash

ollama serve &

sleep 5

ollama pull all-minilm || true
ollama pull llama3.2 || true

streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
