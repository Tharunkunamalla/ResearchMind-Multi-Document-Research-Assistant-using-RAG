import pymupdf

def generate_sample_pdf(filename="sample_paper.pdf"):
    doc = pymupdf.open()

    page = doc.new_page(width=612, height=792)  # Letter size
    
    # Define text elements (text, size, font, vertical offset)
    elements = [
        # Title (y=80)
        ("A Study on Semantic Information Retrieval", 22, "hebo", 80, True),
        # Authors (y=120)
        ("Jane Doe, John Smith, Alice Williams", 11, "helv", 120, False),
        # Affiliations (y=140)
        ("Department of Computer Science, University of Research", 9, "helv", 140, False),
        ("jane.doe@univ.edu, john.smith@univ.edu", 9, "helv", 155, False),
        
        # Abstract Heading (y=200)
        ("Abstract", 12, "hebo", 200, True),
        # Abstract Body (y=220)
        ("This paper presents a novel approach to semantic information retrieval in large-scale document databases. We utilize advanced text representation models and a hybrid search architecture to retrieve documents based on semantic content rather than simple keyword matches. Our experiments demonstrate significant improvements in retrieval performance over keyword-based baselines.", 10, "helv", 220, False),
        
        # Introduction Heading (y=340)
        ("1. Introduction", 12, "hebo", 340, True),
        # Introduction Body (y=360)
        ("Information retrieval systems have evolved from exact keyword matching to semantic search. Modern systems rely on deep language models that encode semantic meanings as vectors in high-dimensional spaces. Despite these advancements, extracting structured sections from scientific papers remains a challenging task due to formatting variations.", 10, "helv", 360, False),
        
        # Methodology Heading (y=470)
        ("2. Proposed Methodology", 12, "hebo", 470, True),
        # Methodology Body (y=490)
        ("Our model consists of three main components: a high-fidelity document parser, a semantic embedding encoder, and a hybrid vector-keyword retrieval engine. We describe the details of the document parser in this section, showcasing how structural cues like font sizes and text blocks are leveraged to map sections dynamically.", 10, "helv", 490, False),
        
        # Results Heading (y=600)
        ("3. Experiments and Results", 12, "hebo", 600, True),
        # Results Body (y=620)
        ("We evaluated our parsing pipeline on a benchmark of 100 computer science research articles. The results show that our font-size-based parser achieves a 95% classification accuracy across major sections, including Abstract, Introduction, and Conclusion. In contrast, standard rule-based tools achieve only 82%.", 10, "helv", 620, False)
    ]
    
    for text, size, font, y, is_bold in elements:
        # We can use insert_textbox to wrap text
        rect = pymupdf.Rect(72, y, 540, y + 100)

        page.insert_textbox(rect, text, fontsize=size, fontname=font, align=0)
        
    # Let's create a second page for Conclusion and References
    page2 = doc.new_page(width=612, height=792)
    elements2 = [
        # Conclusion Heading
        ("4. Conclusion", 12, "hebo", 80, True),
        # Conclusion Body
        ("In this work, we developed a fast and robust PDF parsing engine. The system leverages font sizes and font weight metadata to divide documents into logical sections, paving the way for high-quality RAG pipelines. Future work will investigate table parsing and mathematical equation extraction.", 10, "helv", 100, False),
        
        # References Heading
        ("References", 12, "hebo", 200, True),
        # References List
        ("[1] Doe, J. et al. 'Semantic Web and NLP Systems.' Journal of Web Semantics, 2024.\n[2] Smith, J. and Williams, A. 'Retrieval-Augmented Generation for Scientific Question Answering.' arXiv preprint, 2025.", 10, "helv", 220, False)
    ]
    
    for text, size, font, y, is_bold in elements2:
        rect = pymupdf.Rect(72, y, 540, y + 150)

        page2.insert_textbox(rect, text, fontsize=size, fontname=font, align=0)
        
    doc.save(filename)
    doc.close()
    print(f"Generated sample PDF at: {filename}")

if __name__ == "__main__":
    generate_sample_pdf()
