FROM python:3.11-slim

# Install dependencies for Ollama CLI
RUN apt-get update && \
    apt-get install -y curl unzip && \
    rm -rf /var/lib/apt/lists/*


# Install Ollama CLI (Linux version)
RUN curl -sSL https://ollama.com/install.sh | bash

# Set workdir
WORKDIR /app

RUN pip install streamlit

# Copy Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app


ENV PORT=8501


RUN chmod +x /app/start.sh
# Expose Streamlit port
EXPOSE 8501



# Start Ollama daemon and Streamlit (multi-process)
#CMD bash -c "ollama serve & streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"
CMD ["/app/start.sh"]
