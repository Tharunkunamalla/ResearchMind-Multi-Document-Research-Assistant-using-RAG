import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple
from app.services.embedder import EmbeddingService

class FAISSVectorStore:
    def __init__(self, dimension: int = 384, embedder: EmbeddingService = None):
        self.dimension = dimension
        self.embedder = embedder if embedder is not None else EmbeddingService()
        self.index = faiss.IndexFlatIP(dimension)
        # Maps index ID (int) -> Metadata dictionary
        self.metadata: Dict[int, dict] = {}
        self.current_id = 0

    def add_chunks(self, chunks: List[dict]):
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        # Convert to numpy array and normalize for Cosine Similarity (Inner Product on L2-normalized vectors)
        vectors = np.array(embeddings).astype('float32')
        faiss.normalize_L2(vectors)
        
        # Add to FAISS index
        self.index.add(vectors)
        
        # Map IDs to metadata
        for i, chunk in enumerate(chunks):
            index_id = self.current_id + i
            self.metadata[index_id] = chunk
            
        self.current_id += len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
        if self.index.ntotal == 0:
            return []
            
        query_embedding = self.embedder.embed_text(query)
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search index
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            idx_int = int(idx)
            if idx_int in self.metadata:
                results.append((self.metadata[idx_int], float(score)))
                
        return results

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "index.faiss")
        metadata_path = os.path.join(directory, "metadata.json")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata (convert int keys to string for JSON support)
        serializable_metadata = {str(k): v for k, v in self.metadata.items()}
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "current_id": self.current_id,
                "metadata": serializable_metadata
            }, f, indent=2, ensure_ascii=False)

    def load(self, directory: str) -> bool:
        index_path = os.path.join(directory, "index.faiss")
        metadata_path = os.path.join(directory, "metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return False
            
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.current_id = data.get("current_id", 0)
            raw_metadata = data.get("metadata", {})
            self.metadata = {int(k): v for k, v in raw_metadata.items()}
            
        return True
