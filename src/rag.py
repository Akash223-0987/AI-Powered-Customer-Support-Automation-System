import os
import re
import math
from typing import List, Dict, Any

class BM25Retriever:
    def __init__(self, documents: List[Dict[str, str]], b: float = 0.75, k1: float = 1.5):
        self.b = b
        self.k1 = k1
        self.documents = documents
        self.doc_len = []
        self.doc_freqs = []
        self.avg_doc_len = 0
        self.idf = {}
        self._initialize()

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize by finding words, lowercasing them
        return re.findall(r'\w+', text.lower())

    def _initialize(self):
        total_len = 0
        df = {}
        for doc in self.documents:
            tokens = self._tokenize(doc['content'])
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            
            frequencies = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
                
        self.avg_doc_len = total_len / len(self.documents) if self.documents else 0
        N = len(self.documents)
        for token, freq in df.items():
            # BM25 IDF formula
            self.idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)

    def retrieve(self, query: str, top_k: int = 3, source_filter: List[str] = None) -> List[Dict[str, str]]:
        query_tokens = self._tokenize(query)
        scores = []
        for i, doc in enumerate(self.documents):
            # Apply source filter if provided
            if source_filter and doc['source'] not in source_filter:
                continue
                
            score = 0.0
            freqs = self.doc_freqs[i]
            dl = self.doc_len[i]
            for token in query_tokens:
                if token in freqs:
                    tf = freqs[token]
                    idf = self.idf.get(token, 0.0)
                    numerator = tf * (self.k1 + 1.0)
                    denom_avg = (dl / self.avg_doc_len) if self.avg_doc_len > 0 else 1.0
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * denom_avg)
                    score += idf * (numerator / denominator)
            scores.append((score, doc))
        
        # Sort in descending order
        scores.sort(key=lambda x: x[0], reverse=True)
        # Return docs with score >= 0 (allow context inclusion even with zero matches)
        return [doc for score, doc in scores[:top_k] if score >= 0]

def load_and_chunk_documents(doc_dir: str) -> List[Dict[str, str]]:
    chunks = []
    files = {
        "company_policy.txt": "Company Policy",
        "pricing_guide.txt": "Pricing Guide",
        "technical_manual.txt": "Technical Manual",
        "faq_document.txt": "FAQ Document"
    }
    
    for filename, source_name in files.items():
        filepath = os.path.join(doc_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found.")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if filename == "faq_document.txt":
            # Split FAQs on Q1:, Q2:, etc.
            sections = re.split(r'\n(?=Q\d+:)', content)
            for sec in sections:
                sec_str = sec.strip()
                if sec_str:
                    chunks.append({
                        "source": source_name,
                        "content": sec_str
                    })
        elif filename in ["company_policy.txt", "technical_manual.txt", "pricing_guide.txt"]:
            # Split policy, manual, and pricing guide on numbered section headers (e.g. "1. Refund Policy")
            sections = re.split(r'\n(?=\d+\.\s+[A-Z])', content)
            for sec in sections:
                sec_str = sec.strip()
                if sec_str:
                    chunks.append({
                        "source": source_name,
                        "content": sec_str
                    })
        else:
            sections = content.split('\n\n')
            for sec in sections:
                sec_str = sec.strip()
                if sec_str:
                    chunks.append({
                        "source": source_name,
                        "content": sec_str
                    })
                    
    return chunks

# Singleton for retriever
_retriever = None

def get_retriever(doc_dir: str = "documents") -> BM25Retriever:
    global _retriever
    if _retriever is None:
        chunks = load_and_chunk_documents(doc_dir)
        _retriever = BM25Retriever(chunks)
    return _retriever

def retrieve_context_for_query(query: str, doc_dir: str = "documents", top_k: int = 2, source_filter: List[str] = None) -> str:
    retriever = get_retriever(doc_dir)
    results = retriever.retrieve(query, top_k=top_k, source_filter=source_filter)
    if not results:
        return "No relevant information found in the knowledge base."
    
    formatted_chunks = []
    for r in results:
        formatted_chunks.append(f"[{r['source']}]:\n{r['content']}")
    return "\n\n---\n\n".join(formatted_chunks)
