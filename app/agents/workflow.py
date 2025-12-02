# app/agents/workflow.py
from langgraph.graph import StateGraph, END
from app.agents.state import NewsState
from app.agents.nodes import deduplicate_node, extraction_node, storage_node

# 1. Initialize Graph
workflow = StateGraph(NewsState)

# 2. Add Nodes
workflow.add_node("deduplicate", deduplicate_node)
workflow.add_node("extract", extraction_node)
workflow.add_node("store", storage_node)

# 3. Define Entry Point
workflow.set_entry_point("deduplicate")

# 4. Define Conditional Logic
def route_after_dedup(state: NewsState):
    if state['is_duplicate']:
        return "end_process" # Goes to END
    return "extract"

# 5. Add Edges
workflow.add_conditional_edges(
    "deduplicate",
    route_after_dedup,
    {
        "extract": "extract",
        "end_process": END
    }
)

workflow.add_edge("extract", "store")
workflow.add_edge("store", END)

# 6. Compile
news_agent_app = workflow.compile()