# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from app.agents.workflow import news_agent_app
from app.services.vector_store import vector_service

# This is the "app" variable uvicorn is looking for
app = FastAPI(title="Financial News Intelligence System")

# --- DATA MODELS ---
class ArticleRequest(BaseModel):
    id: str
    text: str
    source: str

class QueryRequest(BaseModel):
    query: str

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "active", "system": "Financial News Multi-Agent System"}

@app.post("/ingest")
def ingest_article(article: ArticleRequest):
    """
    Ingests an article through the LangGraph workflow:
    Deduplication -> Extraction -> Storage
    """
    initial_state = {
        "article_id": article.id,
        "text": article.text,
        "source": article.source,
        "vector": None,
        "is_duplicate": False,
        "duplicate_of": None,
        "entities": {},
        "impact_analysis": []
    }
    
    result = news_agent_app.invoke(initial_state)
    
    return {
        "id": result["article_id"],
        "status": "duplicate" if result["is_duplicate"] else "processed",
        "duplicate_of": result.get("duplicate_of"),
        "entities": result.get("entities"),
        "impact": result.get("impact_analysis")
    }

@app.post("/query")
def search_news(request: QueryRequest):
    """
    RAG Endpoint: Context-Aware Search
    """
    # 1. Convert query to vector
    query_vector = vector_service.embed_text(request.query)
    
    # 2. Search Vector DB
    # We fetch top 5 to capture "Context Expansion"
    results = vector_service.index.query(
        vector=query_vector, 
        top_k=5, 
        include_metadata=True
    )
    
    matches = []
    if results.get('matches'):
        for match in results['matches']:
            matches.append({
                "score": match['score'],
                "text": match['metadata'].get('text'),
                "source": match['metadata'].get('source'),
                "id": match['metadata'].get('id')
            })
        
    return {"query": request.query, "results": matches}