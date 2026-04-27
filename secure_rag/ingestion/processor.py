import os
import shutil
import hashlib
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

class PIIProcessor:
    def __init__(self):
        # Initialize the engine with the spaCy model
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Define which entities to mask
        self.entities = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION"]

    def mask_text(self, text: str) -> str:
        """
        Identifies and masks PII in the given text.
        """
        # 1. Analyze the text for PII
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language='en'
        )

        # 2. Define how to mask (e.g., replace with [PERSON], [EMAIL_ADDRESS])
        operators = {
            entity: OperatorConfig("replace", {"new_value": f"[{entity}]"})
            for entity in self.entities
        }

        # 3. Anonymize the text
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )

        return anonymized_result.text

class DocumentLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load(self) -> List[Document]:
        """
        Load PDF documents from the specified directory.
        """
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return []
            
        loader = DirectoryLoader(
            self.data_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        return loader.load()

class DocumentSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        """
        return self.splitter.split_documents(documents)

class VectorStoreManager:
    def __init__(self, chroma_path: str, embedding_model: str):
        self.chroma_path = chroma_path
        self.embeddings = OllamaEmbeddings(model=embedding_model)

    def calculate_chunk_ids(self, chunks: List[Document]) -> List[str]:
        """
        Calculate unique IDs for document chunks based on metadata.
        Format: source:page:start_index
        """
        last_page_id = None
        current_chunk_index = 0
        ids = []

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Create a unique ID using source, page, and chunk index
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            ids.append(chunk_id)
        
        return ids

    def add_incremental(self, chunks: List[Document]):
        """
        Add documents to Chroma incrementally, avoiding duplicates.
        """
        db = self.get_vector_store()
        
        # Calculate IDs for the new chunks
        chunk_ids = self.calculate_chunk_ids(chunks)
        
        # Get existing IDs from the DB
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Filter out chunks that already exist
        new_chunks = []
        new_ids = []
        for chunk, chunk_id in zip(chunks, chunk_ids):
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

        if len(new_chunks):
            print(f"Adding {len(new_chunks)} new chunks to the database...")
            db.add_documents(new_chunks, ids=new_ids)
            print("Incremental ingestion complete.")
        else:
            print("No new documents to add.")

    def save_to_chroma(self, chunks: List[Document]):
        """
        Create and persist a Chroma vector store from document chunks.
        (Warning: This clears existing data if directory is manually deleted)
        """
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
            
        print(f"Creating embeddings and saving to {self.chroma_path}...")
        chunk_ids = self.calculate_chunk_ids(chunks)
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.chroma_path,
            ids=chunk_ids
        )
    
    def get_vector_store(self):
        """
        Load an existing Chroma vector store.
        """
        return Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings
        )

if __name__ == "__main__":
    # Quick test for PIIProcessor
    processor = PIIProcessor()
    test_text = "My name is Ong Pei Yong. My email is florexong99@gmail.com and I live in Selangor."
    masked = processor.mask_text(test_text)
    print(f"Original: {test_text}")
    print(f"Masked:   {masked}")
