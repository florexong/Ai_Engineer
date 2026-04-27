from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "secure_rag/chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class RAGEngine:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        self.model = ChatOllama(model=LLM_MODEL)
        
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
    def query(self, query_text: str):
        """
        Executes a RAG query.
        """
        # 1. Search DB
        results = self.db.similarity_search_with_relevance_scores(query_text, k=3)
        
        if len(results) == 0:
            return "No relevant context found.", []

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # 2. Generate Answer
        prompt = self.prompt.format(context=context_text, question=query_text)
        
        response = self.model.invoke(prompt)
        response_text = response.content
        
        sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
        return response_text, sources

if __name__ == "__main__":
    # Quick test
    engine = RAGEngine()
    ans, src = engine.query("What is this CV about?")
    print(f"Answer: {ans}")
    print(f"Sources: {src}")
