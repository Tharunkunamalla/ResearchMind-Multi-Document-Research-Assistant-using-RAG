import shutil
import os
from app.services.chunker import DocumentChunker
from app.services.embedder import EmbeddingService
from app.services.vector_store import FAISSVectorStore

def test_chunker_page_mapping():
    print("Testing DocumentChunker page mapping...")
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    
    # Injected page markers in text
    test_text = (
        "[PAGE_NUM:1] This is the text on page one. It is fairly short. "
        "[PAGE_NUM:2] Now we are on page two. The text continues here. We want to check page transitions. "
        "[PAGE_NUM:3] Finally, page three starts with some concluding sentences."
    )
    
    chunks = chunker.split_section("introduction", test_text, "test_paper_123")
    
    print(f"Generated {len(chunks)} chunks:")
    for c in chunks:
        print(f"  ID: {c['chunk_id']} | Page: {c['page_number']} | Text: {c['text']}")
        # Assertions
        assert "[PAGE_NUM:" not in c["text"], "Page markers were not stripped!"
        assert c["paper_id"] == "test_paper_123"
        assert c["section"] == "introduction"
        
    # Check that chunks mapping matches page intervals
    # Chunk 0 should be page 1
    assert chunks[0]["page_number"] == 1, "Page 1 mapping failed"
    # The last chunk should be page 3
    assert chunks[-1]["page_number"] == 3, "Page 3 mapping failed"
    
    print("[SUCCESS] Chunker page mapping verified successfully!\n")
    return chunks

def test_embedder_and_vector_store(chunks):
    print("Testing EmbeddingService and FAISSVectorStore...")
    
    embedder = EmbeddingService()
    vector_store = FAISSVectorStore(embedder=embedder)
    
    # 1. Test embedding generation
    embedding = embedder.embed_text("Test query")
    assert len(embedding) == 384, f"MiniLM dimension should be 384, got {len(embedding)}"
    print("  Embedding dimensions verified (384).")
    
    # 2. Add chunks to FAISS store
    vector_store.add_chunks(chunks)
    assert vector_store.index.ntotal == len(chunks), f"FAISS index should contain {len(chunks)} items, got {vector_store.index.ntotal}"
    print(f"  Added {len(chunks)} vectors to FAISS index.")
    
    # 3. Test persistence
    temp_store_dir = "temp_vector_store"
    vector_store.save(temp_store_dir)
    print("  Saved index and metadata to disk.")
    
    # Load into a new vector store instance
    new_store = FAISSVectorStore(embedder=embedder)
    loaded = new_store.load(temp_store_dir)
    assert loaded == True, "Failed to load index from disk"
    assert new_store.index.ntotal == len(chunks), f"Loaded index size mismatch: {new_store.index.ntotal}"
    print("  Loaded index and metadata from disk successfully.")
    
    # 4. Test search
    query = "concluding sentences page three"
    search_results = new_store.search(query, top_k=2)
    
    print(f"Search results for '{query}':")
    for metadata, score in search_results:
        print(f"  Score: {score:.4f} | Page: {metadata['page_number']} | Text: {metadata['text']}")
        
    # Assert that the most relevant chunk is from page 3 and contains concluding sentences
    top_result = search_results[0][0]
    assert top_result["page_number"] == 3, "Search did not return page 3 chunk as top result!"
    assert "concluding" in top_result["text"].lower(), "Search top result text mismatch"
    
    # Clean up temp store
    if os.path.exists(temp_store_dir):
        shutil.rmtree(temp_store_dir)
        
    print("[SUCCESS] Embedder and FAISS Vector Store verified successfully!\n")

def main():
    chunks = test_chunker_page_mapping()
    test_embedder_and_vector_store(chunks)
    print("All core Phase 2 components verified successfully!")

if __name__ == "__main__":
    main()
