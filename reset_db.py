# reset_db.py
from app.services.vector_store import vector_service

# Delete all vectors in the index
try:
    vector_service.index.delete(delete_all=True)
    print("✅ Pinecone Index cleared successfully!")
except Exception as e:
    print(f"❌ Error clearing index: {e}")