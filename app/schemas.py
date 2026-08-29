from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ParsedDocumentResponse(BaseModel):
    title: str
    authors: List[str]
    sections: Dict[str, str]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    paper_id: str
    page_number: int
    section: str
    score: float
    title: Optional[str] = None

class IndexDocumentResponse(BaseModel):
    status: str
    paper_id: str
    title: str
    chunk_count: int

class Citation(BaseModel):
    citation_id: int
    paper_id: str
    title: str = ""
    section: str
    page_number: int
    score: float
    snippet: str

class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    paper_ids: Optional[List[str]] = None
    system_prompt: Optional[str] = None

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    context_used_count: int

class PaperSummaryRequest(BaseModel):
    paper_id: str

class PaperSummaryResponse(BaseModel):
    paper_id: str
    title: str
    summary: str
    key_findings: List[str]
    sections_summarized: List[str]


