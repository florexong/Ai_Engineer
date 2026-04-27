from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
CHROMA_PATH = "secure_rag/chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

def test_query(query_text: str):
    """
    Search for the query_text in the local ChromaDB.
    """
    # Initialize Ollama Embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Load the vector store
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Search
    print(f"Searching for: '{query_text}'")
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0:
        print("No results found.")
        return

    for doc, score in results:
        print(f"--- Score: {score} ---")
        print(doc.page_content[:200] + "...")
        print(f"Metadata: {doc.metadata}\n")

if __name__ == "__main__":
    test_query("What is this project about?")
