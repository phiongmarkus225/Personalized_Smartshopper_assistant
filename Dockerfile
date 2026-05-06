FROM python:3.10-slim

WORKDIR /app

# Install dependencies lebih dulu (memanfaatkan Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY website/ ./website/
COPY .env .

# Streamlit default port
EXPOSE 8501

# Health check untuk memastikan app berjalan
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Jalankan Streamlit app
CMD ["streamlit", "run", "website/website.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
