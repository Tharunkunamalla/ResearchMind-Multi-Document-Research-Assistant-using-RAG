import shutil
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from app.services.chunker import DocumentChunker
from app.services.embedder import EmbeddingService
from app.services.vector_store import FAISSVectorStore

def test_chunker_page_mapping():
    print("Testing DocumentChunker page mapping...")
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    
    # Each page's text is > chunk_size so it forces a distinct chunk per page
    page1 = "This is the introductory text on page one of our test document. It contains background information. "
    page2 = "Page two continues with the methodology. The approach is described in technical detail here. Good. "
    page3 = "Finally, page three contains the results and conclusion. Experiments show significant improvements. "

    test_text = f"[PAGE_NUM:1] {page1}[PAGE_NUM:2] {page2}[PAGE_NUM:3] {page3}"
    
    chunks = chunker.split_section("introduction", test_text, "test_paper_123")
    
    print(f"Generated {len(chunks)} chunks:")
    for c in chunks:
        print(f"  ID: {c['chunk_id']} | Page: {c['page_number']} | Text: {c['text'][:60]}...")
        # Core assertions
        assert "[PAGE_NUM:" not in c["text"], "Page markers were not stripped!"
        assert c["paper_id"] == "test_paper_123"
        assert c["section"] == "introduction"
        
    # First chunk should start on page 1
    assert chunks[0]["page_number"] == 1, f"Expected page 1 for first chunk, got {chunks[0]['page_number']}"
    # Last chunk should start on page 3
    assert chunks[-1]["page_number"] == 3, f"Expected page 3 for last chunk, got {chunks[-1]['page_number']}"
    # Page numbers should be non-decreasing
    pages = [c["page_number"] for c in chunks]
    assert pages == sorted(pages), f"Page numbers not monotonic: {pages}"
    
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
        
    # The top result should contain relevant text about page 3 / conclusions
    top_result = search_results[0][0]
    top_text = top_result["text"].lower()
    assert any(kw in top_text for kw in ["conclusion", "results", "improvement", "finally", "page three"]), \
        f"Top search result text doesn't seem relevant: {top_result['text']}"
    print(f"  Top result relevant: '{top_result['text'][:60]}...' (page {top_result['page_number']})")

    
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
