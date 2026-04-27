import os
from secure_rag.ingestion.processor import PIIProcessor, DocumentLoader, DocumentSplitter, VectorStoreManager

# Configuration
DATA_PATH = "secure_rag/data"
CHROMA_PATH = "secure_rag/chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

def ingest_docs():
    """
    Process documents, mask PII, and store them in ChromaDB incrementally.
    """
    # 1. Load Documents
    print(f"Loading documents from {DATA_PATH}...")
    loader = DocumentLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages.")

    if not documents:
        print("No documents found to process.")
        return

    # 2. PII Masking (Security First)
    print("Masking PII in documents...")
    pii_processor = PIIProcessor()
    for doc in documents:
        doc.page_content = pii_processor.mask_text(doc.page_content)
    print("PII masking complete.")

    # 3. Split Documents into Chunks
    print("Splitting documents into chunks...")
    splitter = DocumentSplitter()
    chunks = splitter.split(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 4. Create Embeddings and Store in ChromaDB Incrementally
    vsm = VectorStoreManager(CHROMA_PATH, EMBEDDING_MODEL)
    vsm.add_incremental(chunks)
    
    print("Ingestion process complete.")

if __name__ == "__main__":
    ingest_docs()
