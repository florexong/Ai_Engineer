# Disclaimer

I just vibe coding the entire things without even edit nor understand what is going on I just use gemini-cli to complete everything from the folder structure and the design. The only manual stuff I do is init this git repo and publish it.

# Secure Enterprise RAG

> **Note on Development:** This project was built through a rapid, iterative "vibe-coding" process. While the prototype was developed quickly, it is architected on core engineering principles—such as modularity, idempotency, and security-first data handling—to ensure the system is both functional and maintainable.

A local-first, secure Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, and Microsoft Presidio. This project demonstrates secure data ingestion (PII masking) and scalable retrieval.

## 🚀 Key Features
- **Security-First:** Automatic PII masking (names, emails, locations) before data ingestion.
- **Data Engineering Best Practices:** Modular pipeline with incremental, idempotent ingestion.
- **Local Inference:** Fully local AI using Ollama (no data leaves your machine).
- **Production-Ready API:** FastAPI interface for real-time querying.

## 🛠️ Quick Start

### 1. Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- Pull the required models:
  ```bash
  ollama pull llama3
  ollama pull nomic-embed-text
  ```

### 2. Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Ingest Data
Add your PDFs to `secure_rag/data/` and run:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 secure_rag/ingestion/ingest.py
```

### 4. Run API
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 secure_rag/app/main.py
```
*Access the API at `http://localhost:8000`*

## 📁 Project Structure
- `secure_rag/ingestion/`: Pipeline logic (PII masking, loading, vector storage).
- `secure_rag/app/`: FastAPI server and RAG engine.
- `ARCHITECTURE.md`: Detailed system design and flowcharts.
