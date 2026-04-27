from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from secure_rag.app.engine import RAGEngine

app = FastAPI(title="Secure Enterprise RAG API")
engine = RAGEngine()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Secure Enterprise RAG API"}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Endpoint to handle RAG queries.
    """
    try:
        answer, sources = engine.query(request.query)
        return QueryResponse(answer=answer, sources=list(set(sources)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
