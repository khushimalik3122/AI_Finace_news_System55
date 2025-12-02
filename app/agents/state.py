# app/agents/state.py
from typing import TypedDict, List, Optional, Dict, Any

class NewsState(TypedDict):
    article_id: str
    text: str
    source: str
    # Enriched Data
    vector: Optional[List[float]]
    is_duplicate: bool
    duplicate_of: Optional[str] # ID of the original article if duplicate
    entities: Dict[str, List[str]] # {'companies': [], 'sectors': []}
    impact_analysis: List[Dict[str, Any]] # [{'symbol': 'HDFCBANK', 'confidence': 1.0}]