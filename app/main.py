from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import hashlib
import os

from app.schemas import ParsedDocumentResponse, QueryRequest, ChunkResponse, IndexDocumentResponse
from app.parser import PDFParser
from app.services.chunker import DocumentChunker
from app.services.embedder import EmbeddingService
from app.services.vector_store import FAISSVectorStore

app = FastAPI(
    title="AI Research Assistant - Backend",
    description="Backend service for parsing, chunking, and searching research papers.",
    version="2.0.0"
)

# Initialize Services
chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
embedder = EmbeddingService()
vector_store = FAISSVectorStore(embedder=embedder)

VECTOR_STORE_DIR = os.path.join("data", "vector_store")

# Load existing index on startup
if os.path.exists(VECTOR_STORE_DIR):
    loaded = vector_store.load(VECTOR_STORE_DIR)
    if loaded:
        print(f"Loaded existing FAISS index with {vector_store.index.ntotal} vectors.")
    else:
        print("Failed to load existing FAISS index or directory is empty.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "index_vectors_count": vector_store.index.ntotal
    }

@app.post("/api/documents/parse", response_model=ParsedDocumentResponse)
async def parse_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are supported."
        )
    
    try:
        file_bytes = await file.read()
        parser = PDFParser(file_bytes)
        parsed_data = parser.parse()
        return parsed_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while parsing the PDF: {str(e)}"
        )

@app.post("/api/documents/index", response_model=IndexDocumentResponse)
async def index_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are supported."
        )
        
    try:
        file_bytes = await file.read()
        
        # 1. Generate unique paper_id based on file MD5 hash
        hasher = hashlib.md5()
        hasher.update(file_bytes)
        paper_id = hasher.hexdigest()
        
        # 2. Parse PDF text
        parser = PDFParser(file_bytes)
        parsed_data = parser.parse()
        title = parsed_data.get("title", file.filename)
        
        # 3. Check if paper already indexed
        is_indexed = any(c["paper_id"] == paper_id for c in vector_store.metadata.values())
        if is_indexed:
            # Count the existing chunks
            existing_count = sum(1 for c in vector_store.metadata.values() if c["paper_id"] == paper_id)
            return IndexDocumentResponse(
                status="already_indexed",
                paper_id=paper_id,
                title=title,
                chunk_count=existing_count
            )
            
        # 4. Chunk each section separately
        all_chunks = []
        for sec_name, sec_text in parsed_data.get("sections", {}).items():
            chunks = chunker.split_section(sec_name, sec_text, paper_id)
            all_chunks.extend(chunks)
            
        # 5. Embed and add to Vector Store
        if all_chunks:
            vector_store.add_chunks(all_chunks)
            vector_store.save(VECTOR_STORE_DIR)
            
        return IndexDocumentResponse(
            status="success",
            paper_id=paper_id,
            title=title,
            chunk_count=len(all_chunks)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while indexing the PDF: {str(e)}"
        )

@app.post("/api/documents/search", response_model=List[ChunkResponse])
async def search_documents(request: QueryRequest):
    try:
        search_results = vector_store.search(request.query, request.top_k)
        
        response_chunks = []
        for metadata, score in search_results:
            response_chunks.append(ChunkResponse(
                chunk_id=metadata["chunk_id"],
                text=metadata["text"],
                paper_id=metadata["paper_id"],
                page_number=metadata["page_number"],
                section=metadata["section"],
                score=score
            ))
            
        return response_chunks
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during search: {str(e)}"
        )
