from pydantic import BaseModel
from typing import List, Dict

class ParsedDocumentResponse(BaseModel):
    title: str
    authors: List[str]
    sections: Dict[str, str]
