# app/agents/nodes.py
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI  # <--- CHANGED THIS
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from app.services.vector_store import vector_service
from app.agents.state import NewsState

# --- CONFIGURATION ---
# We use Gemini 1.5 Flash - it is fast and efficient for this hackathon
if "GOOGLE_API_KEY" not in os.environ:
    print("⚠️ WARNING: GOOGLE_API_KEY not found in environment!")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0
)

# --- STRUCTURED OUTPUT MODELS ---
class StockImpact(BaseModel):
    symbol: str = Field(description="Stock Ticker Symbol (e.g., RELIANCE, HDFCBANK)")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reason: str = Field(description="Reason for the impact score")

class ExtractionResult(BaseModel):
    companies: List[str] = Field(description="List of companies mentioned")
    sectors: List[str] = Field(description="List of sectors mentioned (e.g., Banking, IT)")
    impacts: List[StockImpact]

# --- NODE 1: DEDUPLICATION AGENT ---
def deduplicate_node(state: NewsState):
    print(f"\n--- [Dedup Agent] Processing: {state['article_id']} ---")
    
    # 1. Generate Embedding
    vector = vector_service.embed_text(state['text'])
    
    # 2. Check Vector DB for duplicates
    # Threshold 0.80 for semantic similarity
    is_dup, metadata = vector_service.search_similar(vector, threshold=0.80)
    
    if is_dup:
        print(f"❌ Duplicate detected! Matches article: {metadata.get('id', 'unknown')}")
        return {
            "is_duplicate": True, 
            "duplicate_of": metadata.get('id'),
            "vector": vector
        }
    
    print("✅ Unique story identified.")
    return {"is_duplicate": False, "vector": vector}

# --- NODE 2: ENTITY EXTRACTION AGENT ---
def extraction_node(state: NewsState):
    print(f"--- [Extraction Agent] Analyzing: {state['article_id']} ---")
    
    system_prompt = """You are a financial analyst AI. Extract entities and map stock impacts.
    RULES:
    1. Identify Company Names and Sectors.
    2. Map to NSE/BSE Ticker Symbols.
    3. Assign Confidence:
       - Direct Mention (e.g., "HDFC Bank buys...") -> 1.0
       - Sector Impact (e.g., "Banking sector rallies...") -> 0.6 to 0.8
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    
    # Bind structured output with Gemini
    chain = prompt | llm.with_structured_output(ExtractionResult)
    
    try:
        result = chain.invoke({"text": state['text']})
        return {
            "entities": {
                "companies": result.companies,
                "sectors": result.sectors
            },
            "impact_analysis": [impact.dict() for impact in result.impacts]
        }
    except Exception as e:
        print(f"⚠️ Extraction Error: {e}")
        # Fallback just in case, though Gemini Free Tier is usually stable
        return {
            "entities": {"companies": [], "sectors": []},
            "impact_analysis": []
        }

# --- NODE 3: STORAGE AGENT ---
def storage_node(state: NewsState):
    if not state['is_duplicate']:
        metadata = {
            "id": state['article_id'],
            "source": state['source'],
            "text": state['text'][:100]
        }
        vector_service.upsert_article(
            state['article_id'], 
            state['vector'], 
            metadata
        )
        print("💾 Saved to Vector DB")
    return {}