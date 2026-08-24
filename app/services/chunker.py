import re
from typing import List, Dict, Tuple

class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_section(self, section_name: str, text: str, paper_id: str) -> List[Dict]:
        """
        Splits a section's text into small semantic chunks, determines their original page numbers,
        and returns list of chunks with metadata.
        """
        if not text or not text.strip():
            return []

        # Find all page markers: [PAGE_NUM:X]
        marker_pattern = r'\[PAGE_NUM:(\d+)\]'
        matches = list(re.finditer(marker_pattern, text))
        
        # Reconstruct clean text (without markers) and map character offsets to page numbers
        clean_text_parts = []
        char_to_page = {}
        last_idx = 0
        current_page = 1  # Default page if no markers found yet
        
        for m in matches:
            start, end = m.span()
            part = text[last_idx:start]
            start_offset = sum(len(p) for p in clean_text_parts)
            clean_text_parts.append(part)
            
            for i in range(len(part)):
                char_to_page[start_offset + i] = current_page
                
            current_page = int(m.group(1))
            last_idx = end
            
        part = text[last_idx:]
        start_offset = sum(len(p) for p in clean_text_parts)
        clean_text_parts.append(part)
        for i in range(len(part)):
            char_to_page[start_offset + i] = current_page
            
        clean_text = "".join(clean_text_parts)
        
        if not clean_text.strip():
            return []

        # Recursively split the clean text
        raw_chunks = self._recursive_split(clean_text, ["\n\n", "\n", ". ", " ", ""], 0)
        
        chunks = []
        for idx, (chunk_txt, start_idx, end_idx) in enumerate(raw_chunks):
            chunk_txt = chunk_txt.strip()
            if not chunk_txt:
                continue
                
            # Retrieve page number (default to current_page if indices are out of map)
            page_number = char_to_page.get(start_idx, current_page)
            
            # Formulate unique chunk id
            # format: {paper_id}_{section}_{index}
            safe_section = re.sub(r'[^a-zA-Z0-9]', '_', section_name)
            chunk_id = f"{paper_id}_{safe_section}_{idx}"
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_txt,
                "paper_id": paper_id,
                "page_number": page_number,
                "section": section_name
            })
            
        return chunks

    def _recursive_split(self, text: str, delimiters: List[str], current_start: int) -> List[Tuple[str, int, int]]:
        """
        Recursively splits text into chunks of target size.
        Returns a list of tuples: (chunk_text, start_char_idx, end_char_idx).
        """
        if len(text) <= self.chunk_size:
            return [(text, current_start, current_start + len(text))]
            
        if not delimiters:
            # Fallback to hard character index chunking if no delimiters left
            chunks = []
            step = self.chunk_size - self.chunk_overlap
            if step <= 0:
                step = self.chunk_size
            for i in range(0, len(text), step):
                chunk = text[i:i + self.chunk_size]
                chunks.append((chunk, current_start + i, current_start + i + len(chunk)))
            return chunks
            
        delim = delimiters[0]
        if delim == "":
            splits = list(text)
        else:
            # We split by delimiter, but we must track offsets.
            # Using re.split with delimiter retention or just splitting and manually tracking indexes:
            splits = text.split(delim)
            
        chunks = []
        buffer = []
        buffer_len = 0
        buffer_start = 0
        
        for i, split in enumerate(splits):
            split_len = len(split)
            delim_len = len(delim) if i < len(splits) - 1 else 0
            
            if split_len > self.chunk_size:
                # Flush the current buffer
                if buffer:
                    buf_text = delim.join(buffer)
                    chunks.append((buf_text, current_start + buffer_start, current_start + buffer_start + len(buf_text)))
                    buffer = []
                    buffer_len = 0
                
                # Split this large block recursively with the next delimiter
                split_start_in_text = text.find(split, buffer_start)
                res = self._recursive_split(split, delimiters[1:], current_start + split_start_in_text)
                chunks.extend(res)
                buffer_start = split_start_in_text + split_len + delim_len
                continue
                
            new_len = buffer_len + split_len + (delim_len if buffer else 0)
            if new_len <= self.chunk_size:
                buffer.append(split)
                buffer_len = new_len
            else:
                # Flush current buffer
                if buffer:
                    buf_text = delim.join(buffer)
                    chunks.append((buf_text, current_start + buffer_start, current_start + buffer_start + len(buf_text)))
                
                # Form overlap buffer
                overlap_buffer = []
                overlap_len = 0
                for prev_split in reversed(buffer):
                    prev_len = len(prev_split)
                    if overlap_len + prev_len + (len(delim) if overlap_buffer else 0) <= self.chunk_overlap:
                        overlap_buffer.insert(0, prev_split)
                        overlap_len += prev_len + (len(delim) if len(overlap_buffer) > 1 else 0)
                    else:
                        break
                        
                buffer = overlap_buffer + [split]
                buffer_len = sum(len(x) for x in buffer) + len(delim) * (len(buffer) - 1)
                
                split_start_in_text = text.find(split, buffer_start)
                buffer_start = split_start_in_text - overlap_len
                
        if buffer:
            buf_text = delim.join(buffer)
            chunks.append((buf_text, current_start + buffer_start, current_start + buffer_start + len(buf_text)))
            
        return chunks
