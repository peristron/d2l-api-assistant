"""
D2L Brightspace API Documentation Assistant
Single-file proof of concept with scraping, embedding, and chat interface.

Usage:
1. First run: python app.py --scrape (crawls docs and builds vector store)
2. Subsequent runs: streamlit run app.py
"""

import streamlit as st
import json
import re
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Generator
from datetime import datetime
from urllib.parse import urljoin, urlparse, urldefrag
import time
import sys

# Third-party imports
import httpx
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL = "https://docs.valence.desire2learn.com/"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "d2l_docs"
MAX_PAGES = 300
CRAWL_DELAY = 0.3

ROUTE_PATTERN = re.compile(r"(GET|POST|PUT|PATCH|DELETE)\s+(/d2l/api/[\w/{}().~\-]+)", re.IGNORECASE)

DEVELOPER_PROMPT = """You are an expert on the Brightspace/D2L Valence API. 
Provide precise technical answers with exact routes, parameters, and JSON examples.
Format responses in Markdown with code blocks."""

PLAIN_PROMPT = """You are a friendly assistant explaining the Brightspace/D2L API.
Use simple language, avoid jargon, and explain things step-by-step.
Format responses in Markdown."""

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Chunk:
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)

# ============================================================================
# SCRAPER
# ============================================================================

class DocCrawler:
    def __init__(self):
        self.visited = set()
        self.pages = []
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "D2L-API-Assistant/1.0"}
        )

    def normalize_url(self, url: str) -> str:
        url, _ = urldefrag(url)
        if url.endswith("/index.html"):
            url = url[:-10]
        return url.rstrip("/")

    def is_valid(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.hostname != "docs.valence.desire2learn.com":
            return False
        skip_ext = {".png", ".jpg", ".gif", ".css", ".js", ".zip", ".pdf"}
        return not any(parsed.path.lower().endswith(ext) for ext in skip_ext)

    def crawl_page(self, url: str):
        try:
            response = self.client.get(url)
            response.raise_for_status()
            
            if "text/html" not in response.headers.get("content-type", ""):
                return None, []

            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            title = soup.find("title")
            title = title.get_text(strip=True) if title else ""
            
            # Find main content
            main = (
                soup.find("div", {"role": "main"}) or
                soup.find("div", class_="document") or
                soup.find("div", class_="rst-content") or
                soup.find("body")
            )
            
            if not main:
                return None, []
            
            content = main.get_text(separator="\n", strip=True)
            
            # Extract category from URL
            path_parts = [p for p in urlparse(url).path.split("/") if p]
            category = path_parts[-1].replace(".html", "") if path_parts else "general"
            
            page_data = {
                "url": url,
                "title": title,
                "content": content,
                "category": category,
                "crawled_at": datetime.utcnow().isoformat()
            }
            
            # Find links
            links = []
            for a in soup.find_all("a", href=True):
                abs_url = urljoin(url, a["href"])
                norm_url = self.normalize_url(abs_url)
                if self.is_valid(norm_url) and norm_url not in self.visited:
                    links.append(norm_url)
            
            return page_data, links
            
        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {e}")
            return None, []

    def crawl_all(self, max_pages=MAX_PAGES):
        queue = [self.normalize_url(BASE_URL)]
        
        while queue and len(self.pages) < max_pages:
            url = queue.pop(0)
            
            if url in self.visited:
                continue
            
            self.visited.add(url)
            logger.info(f"Crawling [{len(self.pages)+1}]: {url}")
            
            page_data, links = self.crawl_page(url)
            
            if page_data and page_data["content"].strip():
                self.pages.append(page_data)
            
            queue.extend(links)
            time.sleep(CRAWL_DELAY)
        
        logger.info(f"Crawled {len(self.pages)} pages")
        return self.pages

    def close(self):
        self.client.close()

# ============================================================================
# CHUNKING & EMBEDDING
# ============================================================================

def create_chunks(pages):
    """Convert pages into searchable chunks."""
    chunks = []
    
    for page in pages:
        content = page["content"]
        url = page["url"]
        title = page["title"]
        category = page["category"]
        
        # Create page summary chunk
        summary = content[:1000]
        chunk_id = hashlib.md5(f"{url}|summary".encode()).hexdigest()[:16]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            content=f"# {title}\nCategory: {category}\n\n{summary}",
            metadata={
                "source_url": url,
                "title": title,
                "category": category,
                "chunk_type": "summary"
            }
        ))
        
        # Extract API routes
        for match in ROUTE_PATTERN.finditer(content):
            method = match.group(1).upper()
            path = match.group(2)
            
            # Get context around the route
            start = max(0, match.start() - 500)
            end = min(len(content), match.end() + 1500)
            route_context = content[start:end]
            
            route_content = (
                f"# API Route: {method} {path}\n"
                f"Page: {title}\n"
                f"Category: {category}\n"
                f"Source: {url}\n\n"
                f"{route_context}"
            )
            
            chunk_id = hashlib.md5(f"{url}|route|{method}|{path}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=route_content,
                metadata={
                    "source_url": url,
                    "title": title,
                    "category": category,
                    "chunk_type": "route",
                    "http_method": method,
                    "api_path": path
                }
            ))
        
        # Split long content into overlapping chunks
        words = content.split()
        chunk_size = 300  # words
        overlap = 50
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text) < 100:  # Skip tiny chunks
                continue
            
            full_content = (
                f"# {title}\n"
                f"Category: {category}\n"
                f"Source: {url}\n\n"
                f"{chunk_text}"
            )
            
            chunk_id = hashlib.md5(f"{url}|content|{i}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=full_content,
                metadata={
                    "source_url": url,
                    "title": title,
                    "category": category,
                    "chunk_type": "content",
                    "part": i // (chunk_size - overlap)
                }
            ))
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks):
    """Build ChromaDB vector store from chunks."""
    Path(CHROMA_DIR).mkdir(exist_ok=True)
    
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Delete existing collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    
    # Add in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        ids = [c.chunk_id for c in batch]
        documents = [c.content for c in batch]
        metadatas = []
        
        for c in batch:
            meta = {}
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)
        
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(f"Added batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
    
    logger.info(f"Vector store built with {collection.count()} chunks")
    return collection

# ============================================================================
# LLM PROVIDERS
# ============================================================================

class HuggingFaceLLM:
    """Free LLM via HuggingFace Inference API."""
    def __init__(self):
        self.model = "mistralai/Mistral-7B-Instruct-v0.3"
        self.base_url = "https://api-inference.huggingface.co/models"
    
    def generate(self, messages, temperature=0.3, max_tokens=2000):
        prompt = self._format_messages(messages)
        
        try:
            response = httpx.post(
                f"{self.base_url}/{self.model}",
                headers={"Content-Type": "application/json"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": max(temperature, 0.01),
                        "return_full_text": False
                    }
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
            
            return "The free model is currently loading. Please try again in a moment."
        except Exception as e:
            logger.error(f"HF API error: {e}")
            return f"Error: {str(e)}"
    
    def _format_messages(self, messages):
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n")
            elif role == "user":
                parts.append(f"{content} [/INST]")
            elif role == "assistant":
                parts.append(f"{content} </s><s>[INST] ")
        return "".join(parts)

class OpenAILLM:
    """OpenAI GPT-4o-mini."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o-mini"
    
    def generate(self, messages, temperature=0.3, max_tokens=2000):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI API error: {e}"
    
    def stream(self, messages, temperature=0.3, max_tokens=2000):
        try:
            with httpx.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True
                },
                timeout=60.0
            ) as response:
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            content = data["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except:
                            continue
        except Exception as e:
            yield f"\n\nError: {e}"

class XaiLLM:
    """xAI Grok."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "grok-3-mini-fast"
        self.base_url = "https://api.x.ai/v1"
    
    def generate(self, messages, temperature=0.3, max_tokens=2000):
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"xAI API error: {e}"
    
    def stream(self, messages, temperature=0.3, max_tokens=2000):
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True
                },
                timeout=60.0
            ) as response:
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            content = data["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except:
                            continue
        except Exception as e:
            yield f"\n\nError: {e}"

# ============================================================================
# RAG ENGINE
# ============================================================================

class RAGEngine:
    def __init__(self):
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    
    def retrieve(self, query, n_results=6):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                chunks.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "relevance": 1 - results["distances"][0][i]
                })
        
        return chunks
    
    def build_prompt(self, query, chunks, persona, history=None):
        system_prompt = DEVELOPER_PROMPT if persona == "developer" else PLAIN_PROMPT
        
        context = "\n\n".join([
            f"--- Source {i+1} [Relevance: {c['relevance']:.2f}] ---\n"
            f"URL: {c['metadata'].get('source_url', 'unknown')}\n"
            f"{c['content']}"
            for i, c in enumerate(chunks)
        ])
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            messages.extend(history[-6:])  # Last 3 turns
        
        user_msg = (
            f"Documentation:\n\n{context}\n\n"
            f"---\n\n"
            f"Question: {query}\n\n"
            f"Answer based on the documentation above."
        )
        
        messages.append({"role": "user", "content": user_msg})
        return messages
    
    def query(self, user_query, llm, persona="developer", history=None):
        chunks = self.retrieve(user_query)
        
        if not chunks:
            return "No relevant documentation found.", []
        
        messages = self.build_prompt(user_query, chunks, persona, history)
        response = llm.generate(messages)
        
        sources = []
        seen = set()
        for c in chunks:
            url = c["metadata"].get("source_url", "")
            if url and url not in seen:
                seen.add(url)
                sources.append({
                    "url": url,
                    "title": c["metadata"].get("title", ""),
                    "relevance": c["relevance"]
                })
        
        return response, sources
    
    def query_stream(self, user_query, llm, persona="developer", history=None):
        chunks = self.retrieve(user_query)
        
        if not chunks:
            def empty():
                yield "No relevant documentation found."
            return empty(), []
        
        messages = self.build_prompt(user_query, chunks, persona, history)
        
        sources = []
        seen = set()
        for c in chunks:
            url = c["metadata"].get("source_url", "")
            if url and url not in seen:
                seen.add(url)
                sources.append({
                    "url": url,
                    "title": c["metadata"].get("title", ""),
                    "relevance": c["relevance"]
                })
        
        return llm.stream(messages), sources

# ============================================================================
# STREAMLIT APP
# ============================================================================

def run_streamlit_app():
    st.set_page_config(
        page_title="D2L API Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 D2L Brightspace API Assistant")
    
    # Check if vector store exists
    if not Path(CHROMA_DIR).exists():
        st.error(
            "⚠️ Knowledge base not found!\n\n"
            "Run: `python app.py --scrape` first to build the knowledge base."
        )
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag" not in st.session_state:
        st.session_state.rag = RAGEngine()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Persona
        persona = st.radio(
            "Response Style:",
            ["developer", "plain_english"],
            format_func=lambda x: "👨‍💻 Developer" if x == "developer" else "📝 Plain English"
        )
        
        st.divider()
        
        # Model selection
        model = st.selectbox(
            "AI Model:",
            ["free", "openai", "xai"],
            format_func=lambda x: {
                "free": "🆓 Free (Mistral 7B)",
                "openai": "🔑 OpenAI GPT-4o-mini",
                "xai": "🔑 xAI Grok"
            }[x]
        )
        
        api_key = None
        if model in ["openai", "xai"]:
            password = st.text_input("Access Password:", type="password")
            if password:
                try:
                    if password == st.secrets.get("MODEL_PASSWORD", ""):
                        st.success("✅ Access granted")
                        api_key = st.secrets.get(
                            "OPENAI_API_KEY" if model == "openai" else "XAI_API_KEY"
                        )
                    else:
                        st.error("❌ Incorrect password")
                        model = "free"
                except:
                    st.warning("No secrets configured")
                    model = "free"
        
        st.divider()
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📖 Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(f"- [{src['title']}]({src['url']}) (relevance: {src['relevance']:.0%})")
    
    # Chat input
    if prompt := st.chat_input("Ask about the D2L API..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get LLM
        if model == "free":
            llm = HuggingFaceLLM()
        elif model == "openai":
            llm = OpenAILLM(api_key)
        else:
            llm = XaiLLM(api_key)
        
        # Build history
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if model in ["openai", "xai"] and hasattr(llm, "stream"):
                    stream, sources = st.session_state.rag.query_stream(
                        prompt, llm, persona, history
                    )
                    response = st.write_stream(stream)
                else:
                    response, sources = st.session_state.rag.query(
                        prompt, llm, persona, history
                    )
                    st.markdown(response)
            
            if sources:
                with st.expander(f"📖 Sources ({len(sources)})"):
                    for src in sources:
                        st.markdown(f"- [{src['title']}]({src['url']}) (relevance: {src['relevance']:.0%})")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

# ============================================================================
# CLI FOR SCRAPING
# ============================================================================

def run_scraper():
    """Run the scraper to build the knowledge base."""
    logger.info("=" * 60)
    logger.info("STARTING SCRAPER")
    logger.info("=" * 60)
    
    # Crawl
    crawler = DocCrawler()
    try:
        pages = crawler.crawl_all(max_pages=MAX_PAGES)
    finally:
        crawler.close()
    
    # Save raw data
    Path("scraped_data.json").write_text(json.dumps(pages, indent=2))
    logger.info(f"Saved {len(pages)} pages to scraped_data.json")
    
    # Create chunks
    chunks = create_chunks(pages)
    
    # Build vector store
    build_vector_store(chunks)
    
    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info(f"  Pages: {len(pages)}")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info("  Run: streamlit run app.py")
    logger.info("=" * 60)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scrape":
        run_scraper()
    else:
        run_streamlit_app()
