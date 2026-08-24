import json
from app.parser import PDFParser
import os
import subprocess

def main():
    pdf_filename = "sample_paper.pdf"
    
    # 1. Generate the sample PDF if it does not exist
    if not os.path.exists(pdf_filename):
        print("Generating sample PDF...")
        subprocess.run(["python", "generate_sample_pdf.py"], check=True)
        
    # 2. Read the PDF bytes
    with open(pdf_filename, "rb") as f:
        file_bytes = f.read()
        
    # 3. Instantiate and run parser
    parser = PDFParser(file_bytes)
    result = parser.parse()
    
    # 4. Print results structured
    print("\n--- Parsed Document Details ---")
    print(f"Title:   {result['title']}")
    print(f"Authors: {result['authors']}")
    print("\n--- Sections Extracted ---")
    for sec_name, content in result['sections'].items():
        print(f"[{sec_name}] ({len(content)} chars):")
        print(f"  {content[:150]}...")
        
    # 5. Assertions to verify correctness
    assert "semantic information retrieval" in result["title"].lower(), "Title mismatch"
    assert any("jane doe" in a.lower() for a in result["authors"]), "Jane Doe missing from authors"
    assert any("john smith" in a.lower() for a in result["authors"]), "John Smith missing from authors"
    assert "abstract" in result["sections"], "Abstract section missing"
    assert "introduction" in result["sections"], "Introduction section missing"
    assert "methodology" in result["sections"], "Methodology section missing"
    assert "experiments" in result["sections"], "Experiments section missing"
    assert "conclusion" in result["sections"], "Conclusion section missing"
    assert "references" in result["sections"], "References section missing"
    
    print("\n[SUCCESS] Parser successfully parsed the document and passed all assertions!")

if __name__ == "__main__":
    main()
