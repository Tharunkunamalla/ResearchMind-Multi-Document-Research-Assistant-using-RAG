import os
import sys

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_full_rag_pipeline():
    print("========================================")
    print("Starting AI Research Assistant RAG Tests")
    print("========================================")

    # 1. Test Health Endpoint
    print("\n1. Testing GET /health...")
    response = client.get("/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    health_data = response.json()
    print(f"  [SUCCESS] Status: {health_data['status']}, Vectors in store: {health_data['index_vectors_count']}")

    # 2. Test Document Parsing Endpoint
    print("\n2. Testing POST /api/documents/parse...")
    with open("sample_paper.pdf", "rb") as f:
        response = client.post(
            "/api/documents/parse",
            files={"file": ("sample_paper.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200, f"Parse endpoint failed: {response.text}"
    parse_data = response.json()
    print(f"  [SUCCESS] Parsed title: {parse_data['title']}")
    print(f"  [SUCCESS] Extracted authors: {parse_data['authors']}")
    print(f"  [SUCCESS] Extracted sections: {list(parse_data['sections'].keys())}")
    assert "semantic information retrieval" in parse_data["title"].lower()
    assert len(parse_data["authors"]) >= 1
    assert "abstract" in parse_data["sections"]

    # 3. Test Invalid File Type Parsing
    print("\n3. Testing POST /api/documents/parse with invalid file type...")
    response = client.post(
        "/api/documents/parse",
        files={"file": ("invalid.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == 400, f"Expected 400 for non-PDF file, got {response.status_code}"
    print("  [SUCCESS] Correctly rejected non-PDF upload with 400 Bad Request.")

    # 4. Test Document Indexing Endpoint
    print("\n4. Testing POST /api/documents/index...")
    with open("sample_paper.pdf", "rb") as f:
        response = client.post(
            "/api/documents/index",
            files={"file": ("sample_paper.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200, f"Index endpoint failed: {response.text}"
    index_data = response.json()
    print(f"  [SUCCESS] Index Status: {index_data['status']}")
    print(f"  [SUCCESS] Paper ID: {index_data['paper_id']}")
    print(f"  [SUCCESS] Chunk Count: {index_data['chunk_count']}")
    assert index_data["status"] in ["success", "already_indexed"]
    assert index_data["chunk_count"] > 0
    paper_id = index_data["paper_id"]

    # 5. Test Duplicate Indexing (Idempotency)
    print("\n5. Testing Duplicate Document Indexing (Idempotency)...")
    with open("sample_paper.pdf", "rb") as f:
        response = client.post(
            "/api/documents/index",
            files={"file": ("sample_paper.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    dup_index_data = response.json()
    assert dup_index_data["status"] == "already_indexed", f"Expected already_indexed, got {dup_index_data['status']}"
    assert dup_index_data["paper_id"] == paper_id
    print("  [SUCCESS] Correctly handled duplicate PDF without creating redundant index entries.")

    # 6. Test Semantic Search Endpoint
    print("\n6. Testing POST /api/documents/search...")
    query = "semantic search and vector embeddings in information retrieval"
    response = client.post(
        "/api/documents/search",
        json={"query": query, "top_k": 3}
    )
    assert response.status_code == 200, f"Search endpoint failed: {response.text}"
    search_results = response.json()
    print(f"  [SUCCESS] Retrieved {len(search_results)} search results for query: '{query}'")
    assert len(search_results) > 0
    
    for i, result in enumerate(search_results):
        print(f"    Result #{i+1}:")
        print(f"      - Chunk ID: {result['chunk_id']}")
        print(f"      - Section: {result['section']}")
        print(f"      - Page: {result['page_number']}")
        print(f"      - Score: {result['score']:.4f}")
        print(f"      - Snippet: {result['text'][:80]}...")
        assert result["paper_id"] == paper_id
        assert result["page_number"] >= 1
        assert "text" in result and len(result["text"]) > 0

    # 7. Test Phase 3 Grounded RAG Query Endpoint
    print("\n7. Testing POST /api/research/query...")
    rag_req = {
        "query": "What classification accuracy did the font-size-based parser achieve?",
        "top_k": 3
    }
    response = client.post("/api/research/query", json=rag_req)
    assert response.status_code == 200, f"RAG query endpoint failed: {response.text}"
    rag_data = response.json()
    print("  [SUCCESS] RAG Answer generated:")
    print(f"    {rag_data['answer']}")
    print(f"  [SUCCESS] Citations returned: {len(rag_data['citations'])}")
    for cit in rag_data["citations"]:
        print(f"    - [Citation {cit['citation_id']}] Section: {cit['section']}, Page: {cit['page_number']}, Score: {cit['score']:.4f}")
    assert rag_data["context_used_count"] > 0
    assert len(rag_data["citations"]) > 0
    assert any("95%" in rag_data["answer"] or "experiments" in c["section"] for c in rag_data["citations"])

    # 8. Test Phase 3 Paper Summarization Endpoint
    print("\n8. Testing POST /api/research/summarize...")
    sum_req = {"paper_id": paper_id}
    response = client.post("/api/research/summarize", json=sum_req)
    assert response.status_code == 200, f"Summarize endpoint failed: {response.text}"
    sum_data = response.json()
    print(f"  [SUCCESS] Paper Title: {sum_data['title']}")
    print(f"  [SUCCESS] Sections Summarized: {sum_data['sections_summarized']}")
    print(f"  [SUCCESS] Key findings: {len(sum_data['key_findings'])}")
    for kf in sum_data["key_findings"]:
        print(f"    * {kf}")
    assert sum_data["paper_id"] == paper_id
    assert len(sum_data["sections_summarized"]) >= 3
    assert len(sum_data["key_findings"]) >= 1

    print("\n========================================================")
    print("[ALL TESTS PASSED] Phases 1, 2, and 3 Verified Cleanly!")
    print("========================================================")

if __name__ == "__main__":
    test_full_rag_pipeline()


