# test_logic.py
import json
from app.agents.workflow import news_agent_app

# Load mock data
with open("data/mock_news.json", "r") as f:
    articles = json.load(f)

# Process the first 5 articles (Includes the HDFC & RBI duplicates)
print("🚀 Starting Batch Processing Test...\n")

for article in articles[:5]:
    initial_state = {
        "article_id": article["id"],
        "text": article["text"],
        "source": article["source"],
        "vector": None,
        "is_duplicate": False,
        "duplicate_of": None,
        "entities": {},
        "impact_analysis": []
    }
    
    # Run the graph
    final_state = news_agent_app.invoke(initial_state)
    
    # Print Results
    print(f"ID: {final_state['article_id']}")
    print(f"Status: {'DUPLICATE ❌' if final_state['is_duplicate'] else 'NEW ✅'}")
    if not final_state['is_duplicate']:
        print(f"Entities: {final_state['entities']}")
        print(f"Impact: {final_state['impact_analysis']}")
    print("-" * 50)