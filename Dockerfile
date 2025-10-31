FROM python:3.12

WORKDIR /fake_news_rag

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]

