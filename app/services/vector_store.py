# app/services/vector_store.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class VectorService:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "financial-news-hackathon"
        
        # Initialize Embeddings (using sentence-transformers as required)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create Index if not exists
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(self.index_name)

    def embed_text(self, text: str):
        return self.embeddings.embed_query(text)

    def search_similar(self, vector, threshold=0.85):
        """
        Checks for duplicates. 
        Returns True if a similar article exists with score > threshold.
        """
        results = self.index.query(
            vector=vector, 
            top_k=1, 
            include_metadata=True
        )
        
        if results['matches']:
            score = results['matches'][0]['score']
            if score >= threshold:
                return True, results['matches'][0]['metadata']
        
        return False, None

    def upsert_article(self, article_id, vector, metadata):
        self.index.upsert(vectors=[(article_id, vector, metadata)])

# Singleton instance for import
vector_service = VectorService()