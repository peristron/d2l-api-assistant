"""
D2L Brightspace API Documentation Assistant
Single-file proof of concept with IN-APP scraping, embedding, and chat.

First run: Automatically scrapes and builds knowledge base (takes ~10 mins)
Subsequent runs: Loads from cache
Admin: Password-protected "Refresh Docs" button to re-scrape
"""

import sys
import os

# ============================================================================
# CRITICAL FIX: SQLITE FOR STREAMLIT CLOUD
# ============================================================================
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import json
import re
import hashlib
import logging
import shutil
import gc
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin, urlparse, urldefrag
from typing import Optional, List, Dict, Any

import httpx
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ============================================================================
# LOGGING SETUP - Enhanced for debugging
# ============================================================================

# Create a custom logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store logs in session state for display
def log_to_session(level: str, message: str):
    """Log message and store in session state for UI display"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] [{level}] {message}"
    
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    
    st.session_state.debug_logs.append(log_entry)
    # Keep only last 100 logs
    st.session_state.debug_logs = st.session_state.debug_logs[-100:]
    
    # Also log to standard logger
    getattr(logger, level.lower(), logger.info)(message)

def log_debug(msg): log_to_session("DEBUG", msg)
def log_info(msg): log_to_session("INFO", msg)
def log_warning(msg): log_to_session("WARNING", msg)
def log_error(msg): log_to_session("ERROR", msg)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL = "https://docs.valence.desire2learn.com/"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "d2l_docs"
SCRAPE_CACHE_FILE = "scrape_metadata.json"
MAX_PAGES = 300
CRAWL_DELAY = 0.2

ROUTE_PATTERN = re.compile(r"(GET|POST|PUT|PATCH|DELETE)\s+(/d2l/api/[\w/{}().~\-]+)", re.IGNORECASE)

DEVELOPER_PROMPT = """You are an expert on the Brightspace/D2L Valence API. 
Provide precise technical answers with exact routes, parameters, and JSON examples.
If the context contains a specific API route, explicitly mention it.
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
        skip_ext = {".png", ".jpg", ".gif", ".css", ".js", ".zip", ".pdf", ".txt"}
        return not any(parsed.path.lower().endswith(ext) for ext in skip_ext)

    def crawl_page(self, url: str):
        try:
            response = self.client.get(url)
            if response.status_code != 200:
                log_warning(f"Skipping {url} (Status {response.status_code})")
                return None, []
            
            if "text/html" not in response.headers.get("content-type", ""):
                return None, []

            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.find("title")
            title = title.get_text(strip=True) if title else "Untitled Page"
            
            main = (
                soup.find("div", {"role": "main"}) or
                soup.find("div", class_="document") or
                soup.find("div", class_="rst-content") or
                soup.find("body")
            )
            
            if not main:
                return None, []
            
            content = main.get_text(separator="\n", strip=True)
            path_parts = [p for p in urlparse(url).path.split("/") if p]
            category = path_parts[0] if path_parts else "general"
            if category.endswith(".html"): 
                category = "general"

            page_data = {
                "url": url,
                "title": title,
                "content": content,
                "category": category,
            }
            
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("mailto:") or href.startswith("#"):
                    continue
                abs_url = urljoin(url, href)
                norm_url = self.normalize_url(abs_url)
                if self.is_valid(norm_url) and norm_url not in self.visited:
                    links.append(norm_url)
            
            return page_data, links
            
        except Exception as e:
            log_warning(f"Failed to crawl {url}: {e}")
            return None, []

    def crawl_all(self, max_pages=MAX_PAGES, progress_callback=None):
        queue = [self.normalize_url(BASE_URL)]
        self.visited.add(queue[0])
        
        while queue and len(self.pages) < max_pages:
            url = queue.pop(0)
            if progress_callback:
                progress_callback(len(self.pages) + 1, url)
            
            log_info(f"Crawling [{len(self.pages)+1}]: {url}")
            page_data, links = self.crawl_page(url)
            
            if page_data and len(page_data["content"]) > 100:
                self.pages.append(page_data)
            
            for link in links:
                if link not in self.visited:
                    self.visited.add(link)
                    queue.append(link)
            
            time.sleep(CRAWL_DELAY)
        
        log_info(f"Crawled {len(self.pages)} pages")
        return self.pages

    def close(self):
        self.client.close()

# ============================================================================
# CHUNKING & EMBEDDING
# ============================================================================

def create_chunks(pages, progress_callback=None):
    chunks = []
    seen_ids = set()
    total_pages = len(pages)
    
    def make_unique_id(base_string):
        chunk_id = hashlib.md5(base_string.encode()).hexdigest()[:16]
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            return chunk_id
        counter = 1
        while f"{chunk_id}_{counter}" in seen_ids:
            counter += 1
        unique_id = f"{chunk_id}_{counter}"
        seen_ids.add(unique_id)
        return unique_id
    
    for idx, page in enumerate(pages):
        if progress_callback:
            progress_callback(idx + 1, total_pages)
        
        content = page["content"]
        url = page["url"]
        title = page["title"]
        category = page["category"]
        
        summary = content[:1000]
        chunk_id = make_unique_id(f"{url}|summary")
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
        
        seen_routes_on_page = set()
        for match in ROUTE_PATTERN.finditer(content):
            method = match.group(1).upper()
            path = match.group(2)
            
            route_key = f"{method}|{path}"
            if route_key in seen_routes_on_page:
                continue
            seen_routes_on_page.add(route_key)
            
            start = max(0, match.start() - 300)
            end = min(len(content), match.end() + 1000)
            route_context = content[start:end]
            
            chunk_id = make_unique_id(f"{url}|route|{method}|{path}")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=f"# API Route: {method} {path}\nPage: {title}\nSource: {url}\n\n{route_context}",
                metadata={
                    "source_url": url,
                    "title": title,
                    "category": category,
                    "chunk_type": "route",
                    "http_method": method,
                    "api_path": path
                }
            ))
        
        words = content.split()
        chunk_size = 300
        overlap = 50
        part_num = 0
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text) < 50: continue
            
            chunk_id = make_unique_id(f"{url}|content|{part_num}")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=f"# {title}\nSource: {url}\nPart: {part_num + 1}\n\n{chunk_text}",
                metadata={
                    "source_url": url,
                    "title": title,
                    "category": category,
                    "chunk_type": "content",
                    "part": part_num
                }
            ))
            part_num += 1
    return chunks

def build_vector_store(chunks, progress_callback=None):
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        try:
            shutil.rmtree(chroma_path)
        except Exception as e:
            log_warning(f"Could not delete folder: {e}")
    
    chroma_path.mkdir(exist_ok=True)
    
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if progress_callback:
            progress_callback(i // batch_size + 1, total_batches)
        
        ids = [c.chunk_id for c in batch]
        documents = [c.content for c in batch]
        metadatas = []
        for c in batch:
            meta = {}
            for k, v in c.metadata.items():
                meta[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v
            metadatas.append(meta)
        
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection

# ============================================================================
# KNOWLEDGE BASE BUILDER
# ============================================================================

def build_knowledge_base(force_rebuild=False):
    cache_path = Path(SCRAPE_CACHE_FILE)
    chroma_path = Path(CHROMA_DIR)
    
    if not force_rebuild and chroma_path.exists() and cache_path.exists():
        try:
            log_info("Loading existing KB...")
            metadata = json.loads(cache_path.read_text())
            client = chromadb.PersistentClient(path=str(chroma_path))
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
            if collection.count() > 0:
                return True, f"Loaded {collection.count()} chunks.", metadata
        except Exception as e:
            log_warning(f"Load failed ({e}), rebuilding...")
    
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    log_container = st.expander("📋 Build Logs", expanded=True)
    
    def log(msg):
        with log_container:
            st.text(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        log_info(msg)
    
    try:
        log("Starting clean build...")
        gc.collect()
        if chroma_path.exists(): shutil.rmtree(chroma_path, ignore_errors=True)
        if cache_path.exists(): cache_path.unlink()
        
        status_placeholder.info("📥 Step 1/3: Crawling Documentation...")
        crawler = DocCrawler()
        pages = crawler.crawl_all(
            max_pages=MAX_PAGES, 
            progress_callback=lambda c, u: progress_bar.progress(min(c / MAX_PAGES * 0.33, 0.33))
        )
        crawler.close()
        
        if not pages: raise Exception("No pages found.")

        status_placeholder.info("✂️ Step 2/3: Chunking content...")
        chunks = create_chunks(
            pages,
            progress_callback=lambda c, t: progress_bar.progress(0.33 + (c / t * 0.33))
        )

        status_placeholder.info("🧠 Step 3/3: Embedding vectors...")
        collection = build_vector_store(
            chunks,
            progress_callback=lambda b, t: progress_bar.progress(0.66 + (b / t * 0.34))
        )
        final_count = collection.count()
        
        metadata = {
            "scraped_at": datetime.utcnow().isoformat(),
            "pages_count": len(pages),
            "chunks_count": len(chunks),
            "vectors_count": final_count,
        }
        cache_path.write_text(json.dumps(metadata, indent=2))
        
        progress_bar.progress(1.0)
        status_placeholder.success("✅ Knowledge base built successfully!")
        return True, "Success", metadata

    except Exception as e:
        status_placeholder.error(f"Build failed: {e}")
        log_error(f"Build failed: {e}")
        return False, str(e), {}

# ============================================================================
# RAG ENGINE
# ============================================================================

class RAGEngine:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
    
    def retrieve(self, query, n_results=5):
        log_debug(f"Retrieving documents for query: {query[:50]}...")
        try:
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
            log_debug(f"Retrieved {len(chunks)} chunks")
            return chunks
        except Exception as e:
            log_error(f"Retrieval error: {e}")
            return []

    def build_messages(self, query, chunks, persona, history=None):
        system_prompt = DEVELOPER_PROMPT if persona == "developer" else PLAIN_PROMPT
        context_text = "\n\n".join([f"--- Source {i+1} ---\n{c['content']}" for i, c in enumerate(chunks)])
        
        messages = [{"role": "system", "content": system_prompt}]
        if history: messages.extend(history[-4:])
        
        user_content = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer using the Context above."
        messages.append({"role": "user", "content": user_content})
        return messages

    def get_sources(self, chunks):
        sources = []
        seen = set()
        for c in chunks:
            url = c["metadata"].get("source_url", "")
            title = c["metadata"].get("title", "Doc")
            if url and url not in seen:
                seen.add(url)
                sources.append({"url": url, "title": title, "relevance": c["relevance"]})
        return sources

# ============================================================================
# LLM PROVIDERS - FIXED VERSIONS
# ============================================================================

class HuggingFaceLLM:
    """Uses Hugging Face Inference API - 2025 working version"""
    
    # These are the only endpoints that work reliably in 2025
    ENDPOINTS = [
        "https://api-inference.huggingface.co/models",
    ]
    
    # Free, high-quality, non-gated models that respond well
    MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "HuggingFaceH4/zephyr-7b-beta",
        "openchat/openchat-3.5-0106",
        "google/gemma-7b-it",
    ]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.working_model = None
        log_info(f"HuggingFaceLLM initialized with key: {api_key[:8]}...")
    
    def generate(self, messages: List[Dict], temperature: float = 0.3) -> str:
        prompt = self._format_messages(messages)
        
        # Try cached working model first
        if self.working_model:
            url = f"https://api-inference.huggingface.co/models/{self.working_model}"
            result = self._call_endpoint(url, prompt, temperature)
            if result and "error" not in result.lower() and "❌" not in result:
                return result
        
        # Try all models
        for model in self.MODELS:
            url = f"https://api-inference.huggingface.co/models/{model}"
            log_info(f"Trying HuggingFace model: {model}")
            result = self._call_endpoint(url, prompt, temperature)
            
            if result and "error" not in result.lower() and len(result.strip()) > 10 and "❌" not in result:
                self.working_model = model
                log_info(f"Success with model: {model}")
                return result
        
        return "❌ No working HuggingFace model found. Try again in a few seconds or use OpenAI/xAI."

    def _call_endpoint(self, url: str, prompt: str, temperature: float) -> str:
        try:
            response = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 1024,
                        "temperature": temperature,
                        "top_p": 0.9,
                        "return_full_text": False,
                        "stop": ["</s>", "<|endoftext|>"]
                    }
                },
                timeout=90.0
            )
            
            log_debug(f"HF response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text", "").strip()
                    if text:
                        return text
                elif isinstance(data, dict):
                    if "generated_text" in data:
                        return data["generated_text"].strip()
                    if "error" in data:
                        log_warning(f"HF Error: {data['error']}")
                        if "loading" in data["error"].lower():
                            return "⏳ Model is loading... try again in 30-60 seconds"
                return str(data)
            
            elif response.status_code == 503:
                return "⏳ Model is loading... try again in 30-60 seconds"
            elif response.status_code == 429:
                return "❌ Rate limited - please wait a moment"
            else:
                log_warning(f"HF {response.status_code}: {response.text[:200]}")
                return f"❌ Error {response.status_code}"
                
        except Exception as e:
            log_error(f"HF exception: {e}")
            return f"❌ Connection error: {e}"

    def stream(self, messages: List[Dict], temperature: float = 0.3):
        # Return full response (free tier has no real streaming)
        yield self.generate(messages, temperature)

    def _format_messages(self, messages: List[Dict]) -> str:
        # Use Mistral/Zephyr style (works with most models)
        formatted = ""
        for m in messages:
            if m["role"] == "system":
                formatted += f"<|system|>\n{m['content']}</s>\n"
            elif m["role"] == "user":
                formatted += f"<|user|>\n{m['content']}</s>\n"
            elif m["role"] == "assistant":
                formatted += f"<|assistant|>\n{m['content']}</s>\n"
        formatted += "<|assistant|>\n"
        return formatted


class OpenAILLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gpt-4o-mini"
        log_info(f"OpenAILLM initialized with key: {api_key[:8]}...")
    
    def generate(self, messages: List[Dict], temperature: float = 0.3) -> str:
        log_debug(f"OpenAI generate called with {len(messages)} messages")
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
                    "max_tokens": 1500
                },
                timeout=60.0
            )
            
            log_debug(f"OpenAI response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                log_error(f"OpenAI error: {response.status_code} - {response.text[:200]}")
                return f"OpenAI Error {response.status_code}: {response.text[:200]}"
                
        except Exception as e:
            log_error(f"OpenAI exception: {e}")
            return f"OpenAI Error: {e}"

    def stream(self, messages: List[Dict], temperature: float = 0.3):
        log_debug("OpenAI stream called")
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
                    "stream": True,
                    "max_tokens": 1500
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
            log_error(f"OpenAI stream error: {e}")
            yield f"Error: {e}"


class XaiLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "grok-beta"
        self.base_url = "https://api.x.ai/v1"
        log_info(f"xAI LLM initialized (model: {self.model})")

    def generate(self, messages: List[Dict], temperature: float = 0.3) -> str:
        log_debug(f"xAI generate called with {len(messages)} messages")
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
                    "max_tokens": 2048
                },
                timeout=90.0
            )
            
            log_debug(f"xAI response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error = response.text[:200]
                log_error(f"xAI error {response.status_code}: {error}")
                return f"xAI Error {response.status_code}: {error}"
                
        except Exception as e:
            log_error(f"xAI exception: {e}")
            return f"xAI connection error: {e}"

    def stream(self, messages: List[Dict], temperature: float = 0.3):
        log_debug("xAI stream called")
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
                    "stream": True,
                    "max_tokens": 2048
                },
                timeout=90.0
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line and line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except:
                            continue
        except Exception as e:
            log_error(f"xAI streaming error: {e}")
            yield f"Streaming error: {e}"


# ============================================================================
# HELPER: Get API Key with logging
# ============================================================================

def get_secret(key_name: str) -> Optional[str]:
    """Safely get a secret with logging"""
    try:
        # Try st.secrets first
        if hasattr(st, 'secrets'):
            # Try direct access
            if key_name in st.secrets:
                value = st.secrets[key_name]
                log_debug(f"Found secret '{key_name}' in st.secrets")
                return value
            
            # Try nested access (e.g., secrets.toml with sections)
            for section in st.secrets:
                if isinstance(st.secrets[section], dict) and key_name in st.secrets[section]:
                    value = st.secrets[section][key_name]
                    log_debug(f"Found secret '{key_name}' in st.secrets[{section}]")
                    return value
        
        # Try environment variable
        value = os.environ.get(key_name)
        if value:
            log_debug(f"Found '{key_name}' in environment variables")
            return value
        
        log_debug(f"Secret '{key_name}' not found")
        return None
        
    except Exception as e:
        log_warning(f"Error getting secret '{key_name}': {e}")
        return None


def list_available_secrets() -> List[str]:
    """List all available secret keys for debugging"""
    secrets = []
    try:
        if hasattr(st, 'secrets'):
            for key in st.secrets:
                if isinstance(st.secrets[key], dict):
                    for subkey in st.secrets[key]:
                        secrets.append(f"{key}.{subkey}")
                else:
                    secrets.append(key)
    except Exception as e:
        log_warning(f"Error listing secrets: {e}")
    return secrets


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="D2L API Helper", page_icon="📚", layout="wide")
    st.title("📚 D2L Brightspace API Assistant")

    # Initialize debug logs
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    
    # Session State Init
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_ready" not in st.session_state:
        st.session_state.kb_ready = False
    if "kb_metadata" not in st.session_state:
        st.session_state.kb_metadata = {}
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    log_debug("Main app started")

    # 1. Initialize Knowledge Base
    if not st.session_state.kb_ready:
        st.info("🔍 Checking knowledge base...")
        success, msg, metadata = build_knowledge_base(force_rebuild=False)
        st.session_state.kb_ready = success
        st.session_state.kb_metadata = metadata
        if not success:
            if st.button("🔄 Try Rebuilding"):
                build_knowledge_base(force_rebuild=True)
                st.rerun()
            st.stop()
        st.rerun()

    # 2. Init RAG
    if "rag" not in st.session_state:
        st.session_state.rag = RAGEngine()
        log_debug("RAG Engine initialized")

    # Sidebar Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if st.session_state.kb_metadata:
            pages = st.session_state.kb_metadata.get('pages_count', '?')
            chunks = st.session_state.kb_metadata.get('chunks_count', '?')
            st.success(f"KB Loaded: {pages} pages / {chunks} chunks")

        persona = st.radio(
            "Response Style:",
            ["developer", "plain"],
            format_func=lambda x: "👨‍💻 Developer" if x == "developer" else "📝 Plain English"
        )
        
        st.divider()
        
        # Model Selection
        model_choice = st.selectbox(
            "Model:",
            ["huggingface", "openai", "xai"],
            format_func=lambda x: {
                "huggingface": "🤗 HuggingFace (Free)",
                "openai": "🧠 GPT-4o-mini (OpenAI)",
                "xai": "🚀 Grok (xAI)"
            }[x]
        )
        
        log_debug(f"Selected model: {model_choice}")
        
        api_key = None
        auth_valid = False
        
        # KEY HANDLING LOGIC - FIXED
        if model_choice == "huggingface":
            # Try to get from secrets first
            api_key = get_secret("HUGGINGFACE_API_KEY") or get_secret("HF_API_KEY")
            
            if api_key:
                st.success("✅ HuggingFace key loaded from secrets")
                auth_valid = True
            else:
                # Ask user for key
                api_key = st.text_input(
                    "HuggingFace API Key:",
                    type="password",
                    help="Get a free key at huggingface.co/settings/tokens"
                )
                if api_key:
                    st.success("✅ Key provided")
                    auth_valid = True
                else:
                    st.warning("⚠️ Please enter your HuggingFace API key")
                    auth_valid = False
                    
        elif model_choice in ["openai", "xai"]:
            # Premium models need password
            pwd = st.text_input("Access Password:", type="password")
            expected_pwd = get_secret("MODEL_PASSWORD") or "password"
            
            if pwd == expected_pwd:
                key_name = f"{model_choice.upper()}_API_KEY"
                api_key = get_secret(key_name)
                
                if api_key:
                    st.success(f"✅ Authenticated - {model_choice.upper()} key loaded")
                    auth_valid = True
                else:
                    st.error(f"❌ {key_name} not found in secrets")
                    auth_valid = False
            elif pwd:
                st.error("❌ Invalid password")
                auth_valid = False
            else:
                st.info("🔒 Enter password for premium models")
                auth_valid = False

        st.divider()
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_query = ""
            log_info("Chat cleared")
            st.rerun()

        # Debug Panel
        with st.expander("🔧 Debug Panel"):
            st.write("**Session State Keys:**")
            st.code(list(st.session_state.keys()))
            
            st.write("**Available Secrets:**")
            secrets_list = list_available_secrets()
            if secrets_list:
                st.code(secrets_list)
            else:
                st.warning("No secrets found")
            
            st.write("**Auth Status:**")
            st.write(f"- model_choice: `{model_choice}`")
            st.write(f"- auth_valid: `{auth_valid}`")
            st.write(f"- api_key exists: `{bool(api_key)}`")
            if api_key:
                st.write(f"- api_key preview: `{api_key[:8]}...`")
            
            st.write("**Recent Logs:**")
            if st.session_state.debug_logs:
                for log_entry in st.session_state.debug_logs[-20:]:
                    st.text(log_entry)
            else:
                st.info("No logs yet")
            
            if st.button("Clear Logs"):
                st.session_state.debug_logs = []
                st.rerun()

        # Admin Tools
        with st.expander("🔐 Admin Tools"):
            admin_pw = st.text_input("Admin Password:", type="password", key="admin_pw")
            expected_admin = get_secret("ADMIN_PASSWORD") or "admin"
            
            if st.button("🔄 Refresh DB"):
                if admin_pw == expected_admin:
                    st.session_state.kb_ready = False
                    if "rag" in st.session_state:
                        del st.session_state.rag
                    build_knowledge_base(force_rebuild=True)
                    st.rerun()
                else:
                    st.error("Invalid admin password")
            
            st.divider()
            
            if st.button("🔍 Diagnose DB"):
                try:
                    client = chromadb.PersistentClient(path=CHROMA_DIR)
                    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
                    
                    data = collection.get(include=["metadatas"])
                    categories = {}
                    urls = set()
                    
                    for m in data["metadatas"]:
                        cat = m.get("category", "unknown")
                        categories[cat] = categories.get(cat, 0) + 1
                        urls.add(m.get("source_url", ""))
                        
                    st.write("### Database Stats")
                    st.write(f"**Total Chunks:** {collection.count()}")
                    st.write(f"**Unique Pages:** {len(urls)}")
                    st.write("**Categories:**")
                    st.json(categories)
                except Exception as e:
                    st.error(f"Error: {e}")

    # Chat Interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(f"- [{s['title']}]({s['url']})")

    # Chat Input
    if prompt := st.chat_input("How do I get a user's enrolled courses?"):
        # Prevent duplicate processing
        if prompt == st.session_state.last_query:
            log_warning("Duplicate query detected, skipping")
        else:
            st.session_state.last_query = prompt
            log_info(f"New query: {prompt[:50]}...")
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Check authentication
            if not auth_valid or not api_key:
                with st.chat_message("assistant"):
                    error_msg = f"⚠️ Please configure authentication for **{model_choice}** in the sidebar."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
                st.stop()

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documentation..."):
                    chunks = st.session_state.rag.retrieve(prompt)
                    sources = st.session_state.rag.get_sources(chunks)
                    log_debug(f"Found {len(chunks)} relevant chunks")
                    
                    # Create LLM instance
                    if model_choice == "openai":
                        llm = OpenAILLM(api_key)
                    elif model_choice == "xai":
                        llm = XaiLLM(api_key)
                    else:
                        llm = HuggingFaceLLM(api_key)
                    
                    # Build conversation
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    messages = st.session_state.rag.build_messages(prompt, chunks, persona, history)
                    
                    log_debug(f"Sending {len(messages)} messages to {model_choice}")
                    
                    # Generate response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    try:
                        for chunk in llm.stream(messages):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")
                        
                        response_placeholder.markdown(full_response)
                        log_info(f"Response generated: {len(full_response)} chars")
                        
                    except Exception as e:
                        full_response = f"❌ Error generating response: {e}"
                        response_placeholder.error(full_response)
                        log_error(f"Generation error: {e}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    })
                    
                    # Show sources
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)})"):
                            for s in sources:
                                st.markdown(f"- [{s['title']}]({s['url']})")


if __name__ == "__main__":
    main()
