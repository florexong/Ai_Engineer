# Architecture: Secure Enterprise RAG

## 1. Design Philosophy: "Data Engineering First"
This project was built to bridge the gap between traditional Data Engineering and modern AI workflows. The design prioritizes **security**, **idempotency**, and **modularity**.

## 2. System Flow

### Ingestion Pipeline (The "Write" Path)
The ingestion process ensures that PII never leaves the local environment in plain text.
```text
[Raw PDF] 
    → [DocumentLoader] 
    → [PIIProcessor (Masks Sensitive Data)] 
    → [DocumentSplitter] 
    → [VectorStoreManager (Incremental Load)] 
    → [ChromaDB]
```
*   **Why:** Masking before embedding ensures that your vector store and LLM context are inherently "Privacy Safe."

### Retrieval Pipeline (The "Read" Path)
```text
[User Query] 
    → [FastAPI Endpoint] 
    → [RAGEngine] 
    → [ChromaDB Retrieval] 
    → [Context-Aware Prompt] 
    → [LLM (llama3)] 
    → [Masked Answer]
```

## 3. Engineering Decisions

| Decision | Approach | Reasoning |
| :--- | :--- | :--- |
| **Security** | Presidio Masking | Prevents PII from reaching the AI context/database. |
| **Idempotency** | Incremental Ingestion | Uses stable IDs (`source:page:index`) to prevent duplicates and minimize redundant compute. |
| **Modularity** | Encapsulated Classes | Separates concerns (loading, masking, storage, generation) for easy future swaps. |
| **Performance** | Local Inference | Using Ollama locally ensures data sovereignty and avoids external API costs. |
