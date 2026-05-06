# SmartShopper Assistant

Repository ini berisi implementasi **Personalized Product Recommendation System** menggunakan Haystack AI, MongoDB Atlas, dan Groq — dibuat untuk materi Bootcamp Data Science dibimbing.id.

---

## Arsitektur Sistem

```
User Query
    │
    ▼
Paraphraser (Groq LLM + Chat Memory)
    │
    ▼
Metadata Filter (Groq LLM → JSON filter)
    │
    ▼
Vector Search (SentenceTransformers + MongoDB Atlas)
    │
    ▼
RAG Generator (Groq LLM + retrieved products)
    │
    ▼
Product Recommendation
```

---

## Prasyarat

- Python 3.10+
- Akun [MongoDB Atlas](https://cloud.mongodb.com) (free tier cukup)
- Akun [Groq](https://console.groq.com) (free tier cukup)
- Docker & Docker Compose (opsional, untuk deployment)

---

## Langkah 1 — Clone & Setup Environment

```bash
# Clone repo
git clone <repo-url>
cd shop_recommendation_dibimbing

# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Langkah 2 — Konfigurasi Environment Variables

```bash
# Salin template .env
cp .env.example .env
```

Buka file `.env` dan isi dengan credentials Anda:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
MONGO_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
```

**Mendapatkan Groq API Key:**
1. Daftar di [console.groq.com](https://console.groq.com)
2. Masuk ke **API Keys** → **Create API Key**
3. Copy key dan paste ke `.env`

**Mendapatkan MongoDB Connection String:**
1. Login ke [cloud.mongodb.com](https://cloud.mongodb.com)
2. Pilih cluster → **Connect** → **Drivers**
3. Copy connection string, ganti `<username>` dan `<password>`

---

## Langkah 3 — Setup MongoDB Atlas

### 3a. Buat Database & Collections

Di MongoDB Atlas, buat database `depato_store` dengan collections:
- `products` — menyimpan produk beserta vector embedding
- `materials` — menyimpan list material (Cotton, Polyester, dst.)
- `categories` — menyimpan list kategori (Tops, Dresses/Jumpsuits, dst.)

### 3b. Jalankan Notebook Store Data

```bash
cd shop_recommendation_dibimbing
jupyter notebook process/store_data.ipynb
```

Notebook ini akan:
1. Load dataset Amazon fashion products dari `data/datasets.pkl`
2. Generate embedding menggunakan SentenceTransformers
3. Simpan semua produk ke MongoDB Atlas

### 3c. Buat Vector Search Index

Di MongoDB Atlas:
1. Masuk ke **Atlas Search** → **Create Search Index**
2. Pilih collection `products`
3. Buat **Vector Search Index** dengan nama `vector_index`:

```json
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

4. Buat **Search Index** (full-text) dengan nama `search_index`:

```json
{
  "mappings": {
    "dynamic": true
  }
}
```

---

## Langkah 4 — Eksplorasi Notebook (Opsional)

Jalankan notebook-notebook berikut secara berurutan untuk memahami setiap komponen:

```bash
# 1. Retrieval dasar dengan vector search
jupyter notebook process/retriever.ipynb

# 2. RAG — retrieval + generation
jupyter notebook process/generator.ipynb

# 3. Metadata filter dengan LLM
jupyter notebook process/generator_filter.ipynb

# 4. Chat memory (conversation history)
jupyter notebook process/chat_memory.ipynb

# 5. AI Agent lengkap
jupyter notebook process/shop_recommendation.ipynb
```

Atau jalankan exercise notebook untuk materi bootcamp:

```bash
jupyter notebook exercise_notebook.ipynb
```

---

## Langkah 5 — Jalankan Streamlit App

```bash
cd shop_recommendation_dibimbing
streamlit run website/website.py
```

Buka browser di [http://localhost:8501](http://localhost:8501)

---

## Langkah 6 — Jalankan FastAPI (REST API)

```bash
cd shop_recommendation_dibimbing
uvicorn website.api:app --host 0.0.0.0 --port 8000 --reload
```

Dokumentasi API tersedia di [http://localhost:8000/docs](http://localhost:8000/docs)

**Endpoints:**

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/health` | Cek status API |
| `POST` | `/recommend` | Kirim query, dapat rekomendasi produk |
| `DELETE` | `/history` | Reset chat history |

**Contoh request:**

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "I want a cotton dress under $50"}'
```

---

## Langkah 7 — Deployment dengan Docker

### Jalankan keduanya sekaligus (UI + API)

```bash
cd shop_recommendation_dibimbing

# Pastikan .env sudah terisi
docker compose up --build
```

Setelah build selesai:
- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- FastAPI: [http://localhost:8000](http://localhost:8000)

### Jalankan satu service saja

```bash
# Hanya Streamlit UI
docker compose up smartshopper-ui

# Hanya FastAPI
docker compose up smartshopper-api
```

### Stop containers

```bash
docker compose down
```

---

## Struktur Repository

```
shop_recommendation_dibimbing/
├── data/
│   └── datasets.pkl            # Dataset Amazon fashion products
├── process/
│   ├── store_data.ipynb        # Langkah 1: Simpan data ke MongoDB
│   ├── retriever.ipynb         # Langkah 2: Vector search
│   ├── generator.ipynb         # Langkah 3: RAG pipeline
│   ├── generator_filter.ipynb  # Langkah 4: Metadata filter
│   ├── chat_memory.ipynb       # Langkah 5: Chat memory
│   └── shop_recommendation.ipynb  # Langkah 6: AI Agent lengkap
├── website/
│   ├── website.py              # Streamlit frontend
│   ├── api.py                  # FastAPI REST API
│   └── template.py             # Jinja2 template untuk metadata filter
├── exercise_notebook.ipynb     # Exercise notebook untuk bootcamp
├── Dockerfile                  # Docker image untuk Streamlit
├── Dockerfile.api              # Docker image untuk FastAPI
├── docker-compose.yml          # Orkestrasi kedua service
├── requirements.txt            # Python dependencies
└── .env.example                # Template environment variables
```

---

## Tech Stack

| Komponen | Library / Service |
|----------|-------------------|
| AI Framework | [Haystack AI](https://haystack.deepset.ai/) |
| LLM | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` |
| Embedding | SentenceTransformers (`all-mpnet-base-v2`) |
| Vector DB | MongoDB Atlas Vector Search |
| Frontend | Streamlit |
| API | FastAPI |
| Deployment | Docker + Docker Compose |

---

## Troubleshooting

**`ModuleNotFoundError: haystack_integrations`**
```bash
pip install groq-haystack mongodb-atlas-haystack
```

**`ServerSelectionTimeoutError` (MongoDB)**
- Pastikan IP Anda di-whitelist di MongoDB Atlas → **Network Access** → **Add IP Address**
- Coba tambahkan `0.0.0.0/0` untuk testing

**`AuthenticationError` (Groq)**
- Pastikan `GROQ_API_KEY` di file `.env` sudah benar
- Cek key masih aktif di [console.groq.com](https://console.groq.com)

**Vector search tidak mengembalikan hasil**
- Pastikan index `vector_index` sudah dibuat dan statusnya **Active** di Atlas Search
- Jalankan ulang `store_data.ipynb` untuk repopulasi data