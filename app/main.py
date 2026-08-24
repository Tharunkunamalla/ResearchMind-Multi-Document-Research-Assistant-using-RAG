from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas import ParsedDocumentResponse
from app.parser import PDFParser

app = FastAPI(
    title="AI Research Assistant - Phase 1: Document Intelligence",
    description="Backend service for parsing and structuring research papers.",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/documents/parse", response_model=ParsedDocumentResponse)
async def parse_pdf(file: UploadFile = File(...)):
    # Verify file extension
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
        # Log exception in actual app
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while parsing the PDF: {str(e)}"
        )
