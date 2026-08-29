import os
import re
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

from app.services.vector_store import FAISSVectorStore
from app.schemas import Citation, RAGQueryResponse, PaperSummaryResponse

load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are ResearchMind, an expert AI Research Assistant specialized in scientific document analysis.\n"
    "Your goal is to answer the user's research query accurately and comprehensively, using ONLY the provided document context.\n\n"
    "Guidelines:\n"
    "1. Strict Grounding: Rely strictly on the provided context. If the context does not contain sufficient information, state clearly what is known and what cannot be answered.\n"
    "2. Precise Citations: Attribute every factual claim to its source using bracketed numbers like [Source 1], [Source 2], referencing specific sections and page numbers when helpful.\n"
    "3. Structure & Clarity: Present findings clearly with scientific precision, using bullet points or paragraphs where appropriate.\n"
    "4. No Hallucinations: Do not fabricate results, metrics, or citations not present in the context."
)

class RAGService:
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
        max_tokens: int = 1024
    ):
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self._groq_client = None
        
        if self.api_key:
            try:
                import httpx
                from groq import Groq
                # Use custom httpx.Client to maintain compatibility across httpx versions
                self._groq_client = Groq(api_key=self.api_key, http_client=httpx.Client())
            except Exception as e:
                print(f"[WARNING] Failed to initialize Groq client: {e}")
                self._groq_client = None

    def _build_context_prompt(self, search_results: List[Tuple[dict, float]]) -> Tuple[str, List[Citation]]:
        """
        Formats retrieved vector chunks into a numbered context block and constructs citation objects.
        """
        context_blocks = []
        citations = []
        
        for idx, (chunk, score) in enumerate(search_results, start=1):
            paper_id = chunk.get("paper_id", "unknown")
            title = chunk.get("title", "Research Document")
            section = chunk.get("section", "general")
            page_num = chunk.get("page_number", 1)
            text = chunk.get("text", "").strip()
            
            snippet = text[:180] + "..." if len(text) > 180 else text
            
            citations.append(Citation(
                citation_id=idx,
                paper_id=paper_id,
                title=title,
                section=section,
                page_number=page_num,
                score=round(score, 4),
                snippet=snippet
            ))
            
            context_blocks.append(
                f"[Source {idx}]\n"
                f"Paper: {title} (ID: {paper_id})\n"
                f"Section: {section} | Page: {page_num}\n"
                f"Content:\n{text}\n"
            )
            
        formatted_context = "\n---\n".join(context_blocks)
        return formatted_context, citations

    def _generate_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Calls the Groq LLM API or falls back to smart extractive synthesis if no API key is set.
        """
        if self._groq_client:
            try:
                response = self._groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[ERROR] Groq API call failed: {e}. Falling back to deterministic synthesis.")
                
        # Local Fallback Synthesis Engine
        return self._local_synthesis_fallback(user_prompt)

    def _local_synthesis_fallback(self, user_prompt: str) -> str:
        """
        Deterministic, grounded synthesis engine used when no external LLM API key is configured.
        Extracts key sentences matching the query from the provided sources and synthesizes a structured answer.
        """
        # Parse query and sources from prompt
        query_match = re.search(r"User Question:\s*(.+?)(?=\n\n|\Z)", user_prompt, re.DOTALL)
        query = query_match.group(1).strip() if query_match else "research query"
        
        source_pattern = r"\[Source (\d+)\]\s*Paper:\s*(.+?)\s*\(ID:\s*(.+?)\)\s*Section:\s*(.+?)\s*\|\s*Page:\s*(\d+)\s*Content:\s*(.+?)(?=\n---\n|\Z)"
        sources = list(re.finditer(source_pattern, user_prompt, re.DOTALL))
        
        if not sources:
            return "No relevant research context was found in the indexed documents to answer this question."
            
        answer_paragraphs = [
            f"Based on the indexed research documents, here is the synthesized answer regarding **{query}**:\n"
        ]
        
        for m in sources:
            src_num = m.group(1)
            title = m.group(2).strip()
            sec = m.group(4).strip()
            page = m.group(5).strip()
            content = m.group(6).strip().replace("\n", " ")
            
            # Extract most informative sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if len(s.strip()) > 20]
            top_sentences = sentences[:2] if sentences else [content[:200]]
            bullet_text = " ".join(top_sentences)
            
            answer_paragraphs.append(
                f"- **{sec.capitalize()} (Page {page})**: {bullet_text} [Source {src_num}]"
            )
            
        answer_paragraphs.append(
            "\n*All findings above are cited directly from the indexed document sections.*"
        )
        return "\n".join(answer_paragraphs)

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        paper_ids: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None
    ) -> RAGQueryResponse:
        """
        Executes end-to-end RAG question answering:
        1. Retrieves relevant chunks via FAISS vector search.
        2. Assembles structured context and citation metadata.
        3. Prompts LLM for grounded answer with source citations.
        """
        search_results = self.vector_store.search(query, top_k=top_k, paper_ids=paper_ids)
        
        if not search_results:
            return RAGQueryResponse(
                query=query,
                answer="No relevant documents or sections found in the vector database matching your query. Please ensure papers are indexed.",
                citations=[],
                context_used_count=0
            )
            
        formatted_context, citations = self._build_context_prompt(search_results)
        
        system_prompt = custom_system_prompt or DEFAULT_SYSTEM_PROMPT
        user_prompt = (
            f"Context Documents:\n"
            f"{formatted_context}\n\n"
            f"User Question: {query}\n\n"
            f"Provide a direct, thorough, and accurately cited scientific response to the user question based ONLY on the context above."
        )
        
        answer = self._generate_llm_response(system_prompt, user_prompt)
        
        return RAGQueryResponse(
            query=query,
            answer=answer,
            citations=citations,
            context_used_count=len(citations)
        )

    def summarize_paper(self, paper_id: str) -> PaperSummaryResponse:
        """
        Generates an executive summary of a specific paper by aggregating its indexed sections.
        """
        chunks = self.vector_store.get_chunks_by_paper_id(paper_id)
        if not chunks:
            raise ValueError(f"Paper with ID '{paper_id}' not found in vector store.")
            
        title = chunks[0].get("title", "Research Document")
        
        # Group text by section
        sections_map = {}
        for c in chunks:
            sec = c.get("section", "general")
            if sec not in sections_map:
                sections_map[sec] = []
            sections_map[sec].append(c.get("text", ""))
            
        sections_summarized = list(sections_map.keys())
        
        # Build prompt for summarization
        aggregated_content = []
        for sec, texts in sections_map.items():
            combined = " ".join(texts)
            aggregated_content.append(f"### Section: {sec.upper()}\n{combined}")
            
        full_paper_text = "\n\n".join(aggregated_content)
        
        system_prompt = (
            "You are an expert research synthesizer. Given the sections of a scientific research paper, "
            "provide a concise executive summary and a list of key findings."
        )
        user_prompt = (
            f"Paper Title: {title}\n\n"
            f"Paper Content:\n{full_paper_text}\n\n"
            f"Please generate:\n"
            f"1. A comprehensive 2-3 paragraph executive summary of the paper's core contributions and methodology.\n"
            f"2. A bulleted list of 3-5 key findings / takeaways."
        )
        
        summary_text = self._generate_llm_response(system_prompt, user_prompt)
        
        # Extract key findings if possible, or build structured bullets
        key_findings = []
        findings_matches = re.findall(r"(?:-|\*|\d+\.)\s+(.+)", summary_text)
        if findings_matches:
            key_findings = [f.strip() for f in findings_matches if len(f.strip()) > 15][:5]
        else:
            # Fallback key findings from section extracts
            for sec in ["abstract", "methodology", "experiments", "conclusion"]:
                if sec in sections_map and sections_map[sec]:
                    snippet = sections_map[sec][0].strip()[:150]
                    key_findings.append(f"{sec.capitalize()}: {snippet}...")
                    
        return PaperSummaryResponse(
            paper_id=paper_id,
            title=title,
            summary=summary_text,
            key_findings=key_findings,
            sections_summarized=sections_summarized
        )
