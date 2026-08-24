from pydantic import BaseModel
from typing import List, Dict

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

class IndexDocumentResponse(BaseModel):
    status: str
    paper_id: str
    title: str
    chunk_count: int

