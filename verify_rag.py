import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from app.services.chunker import DocumentChunker
from app.services.embedder import EmbeddingService
from app.services.vector_store import FAISSVectorStore
from app.services.rag_service import RAGService

def test_rag_service():
    print("========================================")
    print("Testing Phase 3: RAG Service & Synthesis")
    print("========================================")

    # 1. Initialize components
    embedder = EmbeddingService()
    vector_store = FAISSVectorStore(embedder=embedder)
    rag_service = RAGService(vector_store=vector_store)

    paper_id = "test_paper_ai_001"
    paper_title = "Advances in Dense Retrieval for Scientific QA"

    # 2. Add sample chunks
    sample_chunks = [
        {
            "chunk_id": f"{paper_id}_abstract_0",
            "text": "In this paper, we propose a dense retrieval model trained on 50,000 scientific abstracts. Our model achieves a Recall@5 of 92.4% on biomedical benchmarks.",
            "paper_id": paper_id,
            "title": paper_title,
            "page_number": 1,
            "section": "abstract"
        },
        {
            "chunk_id": f"{paper_id}_methodology_0",
            "text": "The architecture incorporates bi-encoder transformers with contrastive learning and cross-entropy loss over hard negatives to produce 384-dimensional embeddings.",
            "paper_id": paper_id,
            "title": paper_title,
            "page_number": 2,
            "section": "methodology"
        },
        {
            "chunk_id": f"{paper_id}_experiments_0",
            "text": "We evaluate on three distinct datasets: BioASQ, SciFact, and TREC-COVID. Dense retrieval outperforms BM25 keyword matching by 14.8% in MRR@10.",
            "paper_id": paper_id,
            "title": paper_title,
            "page_number": 3,
            "section": "experiments"
        },
        {
            "chunk_id": f"{paper_id}_conclusion_0",
            "text": "We demonstrated that dense embeddings significantly improve scientific literature search accuracy. In future work, we plan to extend the approach to multimodal chart and table reasoning.",
            "paper_id": paper_id,
            "title": paper_title,
            "page_number": 4,
            "section": "conclusion"
        }
    ]

    vector_store.add_chunks(sample_chunks)
    print(f"  Indexed {len(sample_chunks)} test chunks into vector store.")

    # 3. Test Question Answering
    query = "What benchmark recall did the proposed dense retrieval model achieve?"
    print(f"\n1. Testing RAG Answer for query: '{query}'")
    response = rag_service.answer_query(query=query, top_k=2)

    print("\n--- Generated Answer ---")
    print(response.answer)
    print("\n--- Citations Generated ---")
    for cit in response.citations:
        print(f"  [Citation #{cit.citation_id}] Doc: {cit.title} | Sec: {cit.section} | Page: {cit.page_number} | Score: {cit.score:.4f}")
        print(f"    Snippet: {cit.snippet}")

    assert response.context_used_count == 2
    assert len(response.citations) == 2
    assert response.citations[0].paper_id == paper_id
    assert "92.4%" in response.answer or "abstract" in response.answer.lower()
    print("  [SUCCESS] RAG question answering and citation attribution verified!")

    # 4. Test Scoped Query by Paper ID
    print("\n2. Testing Scoped Query Filtering...")
    scoped_response = rag_service.answer_query(
        query="What is the architecture and embedding dimension?",
        top_k=2,
        paper_ids=[paper_id]
    )
    assert scoped_response.context_used_count > 0
    assert all(c.paper_id == paper_id for c in scoped_response.citations)
    print("  [SUCCESS] Scoped paper_ids query filtering verified!")

    # 5. Test Paper Summarization
    print("\n3. Testing Paper Summarization...")
    summary_resp = rag_service.summarize_paper(paper_id=paper_id)
    print(f"  Summary for: {summary_resp.title}")
    print(f"  Sections summarized: {summary_resp.sections_summarized}")
    print(f"  Key Findings count: {len(summary_resp.key_findings)}")
    for kf in summary_resp.key_findings:
        print(f"    * {kf}")
    
    assert summary_resp.paper_id == paper_id
    assert len(summary_resp.sections_summarized) >= 3
    assert len(summary_resp.key_findings) > 0
    print("  [SUCCESS] Paper summarization verified!")

    print("\n========================================================")
    print("All Phase 3 RAG Service Components Verified Successfully!")
    print("========================================================")

if __name__ == "__main__":
    test_rag_service()
