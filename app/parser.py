import pymupdf  # PyMuPDF
import re
import unicodedata
from typing import Dict, List, Tuple

class PDFParser:
    def __init__(self, file_bytes: bytes):
        self.file_bytes = file_bytes
        self.doc = None


    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)
        
        # Explicit ligature replacements
        ligatures = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "ﬆ": "st",
            "Œ": "OE",
            "œ": "oe",
            "Æ": "AE",
            "æ": "ae"
        }
        for lig, rep in ligatures.items():
            text = text.replace(lig, rep)
            
        # Rejoin hyphenated words at line breaks (e.g., "semanti-\ncally" -> "semantically")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Replace single newlines with spaces, but keep double newlines (paragraphs)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        # Replace multiple spaces/tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()

    def parse(self) -> dict:
        self.doc = pymupdf.open(stream=self.file_bytes, filetype="pdf")
        
        # 1. Determine body font size
        body_font_size = self._determine_body_font_size()
        
        # 2. Extract Title and Authors from page 1
        title, authors = self._extract_title_and_authors(body_font_size)
        
        # 3. Extract sections across all pages
        sections = self._extract_sections(body_font_size)
        
        self.doc.close()
        
        return {
            "title": title,
            "authors": authors,
            "sections": sections
        }

    def _determine_body_font_size(self) -> float:
        sizes = {}
        # Scan first few pages to estimate body text font size
        for page_idx in range(min(3, len(self.doc))):
            page = self.doc[page_idx]
            try:
                blocks = page.get_text("dict")["blocks"]
            except Exception:
                continue
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            size = round(s["size"], 1)
                            # ignore extremely small text (likely noise/subscripts)
                            if size < 5.0:
                                continue
                            sizes[size] = sizes.get(size, 0) + len(s["text"])
        
        if not sizes:
            return 10.0  # default fallback
        
        # Return the font size with the highest character count (mode)
        body_size = max(sizes, key=sizes.get)
        return body_size

    def _extract_title_and_authors(self, body_font_size: float) -> Tuple[str, List[str]]:
        page = self.doc[0]
        try:
            blocks = page.get_text("dict")["blocks"]
        except Exception:
            return "Unknown Title", []
        
        # Collect all text spans with their metadata
        spans = []
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        text = s["text"].strip()
                        if not text:
                            continue
                        spans.append({
                            "text": text,
                            "size": s["size"],
                            "bbox": s["bbox"], # (x0, y0, x1, y1)
                            "font": s["font"],
                            "flags": s["flags"]
                        })
        
        if not spans:
            return "Unknown Title", []

        first_page_height = page.rect.height
        
        # Find Title Candidates:
        # Title must be in upper half of page 1 and larger than body font size + 1.5
        candidate_title_spans = [
            s for s in spans 
            if s["bbox"][1] < first_page_height * 0.5
            and s["size"] > body_font_size + 1.5
        ]
        
        title = ""
        title_y_end = first_page_height * 0.3 # Fallback boundary for authors
        
        if candidate_title_spans:
            max_size = max(s["size"] for s in candidate_title_spans)
            
            # The title spans are those with font size within 1.5pt of the max font size
            title_spans = [
                s for s in candidate_title_spans 
                if abs(s["size"] - max_size) < 1.5
            ]
            title_spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
            
            # Group spans that are on the same line (similar y-coordinate)
            lines = []
            current_line = []
            last_y = -1
            for s in title_spans:
                if last_y == -1 or abs(s["bbox"][1] - last_y) < 5:
                    current_line.append(s["text"])
                else:
                    lines.append(" ".join(current_line))
                    current_line = [s["text"]]
                last_y = s["bbox"][1]
            if current_line:
                lines.append(" ".join(current_line))
            
            title = " ".join(lines)
            title = self.clean_text(title)
            
            title_y_end = max(s["bbox"][3] for s in title_spans)
        else:
            # Fallback to the first few lines of the page
            spans.sort(key=lambda s: s["bbox"][1])
            fallback_title_spans = spans[:3]
            title = " ".join([s["text"] for s in fallback_title_spans])
            title = self.clean_text(title)
            if spans:
                title_y_end = max(s["bbox"][3] for s in fallback_title_spans)
        
        # Find the Abstract start to act as a lower bound for authors
        abstract_y_start = first_page_height * 0.8
        for s in spans:
            txt_lower = s["text"].lower().strip()
            if "abstract" in txt_lower and len(txt_lower) < 35:
                abstract_y_start = s["bbox"][1]
                break
                
        # Find Author Candidates:
        # Located between title_y_end and abstract_y_start
        author_candidate_spans = [
            s for s in spans
            if title_y_end < s["bbox"][1] < abstract_y_start
            and (body_font_size - 1.0 <= s["size"] < (body_font_size + 5.0) or "bold" in s["font"].lower())
        ]
        
        author_lines = {}
        for s in author_candidate_spans:
            y_coord = round(s["bbox"][1], 1)
            matched_y = None
            for existing_y in author_lines.keys():
                if abs(existing_y - y_coord) < 4:
                    matched_y = existing_y
                    break
            if matched_y is not None:
                author_lines[matched_y].append(s)
            else:
                author_lines[y_coord] = [s]
                
        sorted_y_coords = sorted(author_lines.keys())
        authors = []
        
        exclusion_patterns = [
            r'@', r'email', r'department', r'university', r'school', r'institute', 
            r'college', r'laboratory', r'center', r'science', r'engineering', 
            r'abstract', r'introduction', r'http', r'www', r'\.org', r'\.edu', r'\.com'
        ]
        
        def is_likely_name(name_str: str) -> bool:
            words = name_str.strip().split()
            if not words or len(words) > 4:
                return False
            lowercase_exceptions = {"and", "de", "van", "der", "von", "di", "y", "et", "al", "la", "le", "of"}
            capitalized_words = 0
            for w in words:
                w_clean = re.sub(r'[^\w]', '', w)
                if not w_clean:
                    continue
                if w_clean.lower() in lowercase_exceptions:
                    continue
                if w_clean[0].isupper():
                    capitalized_words += 1
                else:
                    return False
            return capitalized_words > 0

        for y in sorted_y_coords:
            line_spans = sorted(author_lines[y], key=lambda s: s["bbox"][0])
            line_text = " ".join([s["text"] for s in line_spans])
            
            # Stop parsing authors as soon as we hit an affiliation or abstract section
            if any(re.search(pat, line_text, re.IGNORECASE) for pat in exclusion_patterns):
                break
                
            if len(line_text) > 120:
                continue
                
            # Remove affiliation superscripts
            cleaned_line = re.sub(r'[\d\*\†\‡\§\u2020\u2021\u002a]+', '', line_text)
            
            # Split by comma, "and", or semi-colon
            parts = re.split(r',|\band\b|;', cleaned_line)
            for p in parts:
                name = p.strip()
                if name and len(name) > 2 and len(name) < 40:
                    name = re.sub(r'\s+', ' ', name)
                    if name.lower() not in ["abstract", "keywords", "key words", "preprint", "version", "proceedings"]:
                        if is_likely_name(name):
                            authors.append(name)
                        
        seen = set()
        dedup_authors = []
        for a in authors:
            if a.lower() not in seen:
                seen.add(a.lower())
                dedup_authors.append(a)
                
        return title, dedup_authors


    def _is_heading(self, block_text: str, first_span_size: float, body_font_size: float, is_bold: bool) -> bool:
        text = block_text.strip()
        if not text:
            return False
            
        word_count = len(text.split())
        if word_count > 15:
            return False
            
        clean_t = re.sub(r'[\d\.\s\-\:]+', '', text).lower()
        
        standard_sections = {
            "abstract", "introduction", "methodology", "methods", "experiments", 
            "results", "conclusion", "conclusions", "references", "bibliography", 
            "relatedwork", "discussion", "background", "evaluation", "futurework", 
            "implementation", "acknowledgments", "acknowledgement", "appendix", 
            "appendices", "overview", "systemarchitecture"
        }
        
        if clean_t in standard_sections:
            return True
            
        is_large = first_span_size >= body_font_size + 1.0
        is_bold_heading = is_bold and first_span_size >= body_font_size + 0.2
        
        if is_large or is_bold_heading:
            heading_patterns = [
                r'^(?:[I|V|X\d]+\.?\s+)+[A-Z]',
                r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',
                r'^[A-Z\s]+$',
            ]
            if any(re.match(pat, text) for pat in heading_patterns):
                if not re.search(r'[a-z]\.$', text):
                    return True
                    
        return False

    def _is_header_or_footer(self, text: str, bbox: Tuple[float, float, float, float], page_height: float) -> bool:
        t = text.strip()
        if not t:
            return True
            
        if re.match(r'^\d+$', t):
            return True
            
        y0, y1 = bbox[1], bbox[3]
        if y0 < 45 or y1 > page_height - 45:
            if len(t) < 80:
                return True
                
        return False

    def _normalize_section_name(self, heading: str) -> str:
        h = heading.lower().strip()
        
        if "abstract" in h:
            return "abstract"
        if "introduction" in h or "intro" in h or "background" in h or "overview" in h:
            return "introduction"
        if "method" in h or "model" in h or "approach" in h or "framework" in h or "algorithm" in h or "architecture" in h:
            return "methodology"
        if "experiment" in h or "evaluation" in h:
            return "experiments"
        if "result" in h or "discussion" in h or "findings" in h:
            return "results"
        if "conclusion" in h or "summary" in h:
            return "conclusion"
        if "reference" in h or "bibliography" in h or "citations" in h:
            return "references"
        if "related work" in h or "prior work" in h or "literature" in h:
            return "related work"
            
        cleaned_key = re.sub(r'[^a-z0-9\s_\-]', '', h)
        cleaned_key = re.sub(r'[\s\-]+', ' ', cleaned_key).strip()
        return cleaned_key

    def _extract_sections(self, body_font_size: float) -> Dict[str, str]:
        sections = {}
        current_section = "preface"
        section_buffer = []
        
        numbering_regex = r'^(?:[I|V|X\d]+\.?\s+)+|^(?:[A-Z]\.\s+)'
        
        for page_idx in range(len(self.doc)):
            page = self.doc[page_idx]
            try:
                blocks = page.get_text("dict")["blocks"]
            except Exception:
                continue
            
            page_marker_needed = True
            
            for b in blocks:
                if "lines" not in b:
                    continue
                    
                block_text_lines = []
                first_span_size = body_font_size
                is_bold = False
                
                if b["lines"] and b["lines"][0]["spans"]:
                    first_span = b["lines"][0]["spans"][0]
                    first_span_size = first_span["size"]
                    font_name = first_span["font"].lower()
                    is_bold = "bold" in font_name or "black" in font_name or first_span["flags"] & 16
                    
                for l in b["lines"]:
                    line_text = "".join([s["text"] for s in l["spans"]])
                    block_text_lines.append(line_text)
                    
                block_text = " ".join(block_text_lines).strip()
                if not block_text:
                    continue
                    
                if self._is_heading(block_text, first_span_size, body_font_size, is_bold):
                    if section_buffer:
                        text_content = self.clean_text(" ".join(section_buffer))
                        if text_content:
                            if current_section in sections:
                                sections[current_section] += " " + text_content
                            else:
                                sections[current_section] = text_content
                        section_buffer = []
                    
                    cleaned_heading = re.sub(numbering_regex, '', block_text).strip()
                    current_section = self._normalize_section_name(cleaned_heading)
                    # The heading shifts us to a new section, which will need a page marker for its first block
                    page_marker_needed = True
                else:
                    if self._is_header_or_footer(block_text, b["bbox"], page.rect.height):
                        continue
                        
                    if page_marker_needed:
                        section_buffer.append(f"[PAGE_NUM:{page_idx + 1}]")
                        page_marker_needed = False
                        
                    section_buffer.append(block_text)
                    
        if section_buffer:
            text_content = self.clean_text(" ".join(section_buffer))
            if text_content:
                if current_section in sections:
                    sections[current_section] += " " + text_content
                else:
                    sections[current_section] = text_content
                    
        if "preface" in sections and len(sections["preface"]) < 100:
            del sections["preface"]
            
        return sections
