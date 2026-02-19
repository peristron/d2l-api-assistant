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

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_to_session(level: str, message: str):
    """Log message and store in session state for UI display"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] [{level}] {message}"
    
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    
    st.session_state.debug_logs.append(log_entry)
    st.session_state.debug_logs = st.session_state.debug_logs[-100:]
    
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
# LLM PROVIDERS
# ============================================================================

class HuggingFaceLLM:
    """Uses Hugging Face Inference API - 2025 working version with extended model list"""
    
    # Extended list of models to try - some may be loaded, others may need warm-up
    MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "HuggingFaceH4/zephyr-7b-beta",
        "HuggingFaceH4/zephyr-7b-alpha",
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen2-7B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "tiiuae/falcon-7b-instruct",
        "openchat/openchat-3.5-0106",
    ]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.working_model = None
        if api_key:
            log_info(f"HuggingFaceLLM initialized with key: {api_key[:10]}...")
        else:
            log_error("HuggingFaceLLM initialized WITHOUT key!")
    
    def generate(self, messages: List[Dict], temperature: float = 0.3) -> str:
        if not self.api_key:
            return "❌ No HuggingFace API key provided"
            
        prompt = self._format_messages(messages)
        log_debug(f"Prompt length: {len(prompt)} characters")
        
        # Try cached working model first
        if self.working_model:
            log_info(f"Trying cached model: {self.working_model}")
            result = self._call_model(self.working_model, prompt, temperature)
            if result and not result.startswith("❌") and not result.startswith("⏳"):
                return result
            log_warning(f"Cached model failed, trying others...")
        
        # Try all models
        errors = []
        for model in self.MODELS:
            log_info(f"Trying model: {model}")
            result = self._call_model(model, prompt, temperature)
            
            if result and not result.startswith("❌") and not result.startswith("⏳") and len(result.strip()) > 20:
                self.working_model = model
                log_info(f"✅ Success with model: {model}")
                return result
            else:
                errors.append(f"{model.split('/')[-1]}: {result[:50] if result else 'No response'}")
                log_warning(f"Model {model} failed: {result[:100] if result else 'No response'}")
        
        # All failed
        error_summary = "\n".join(errors[-5:])
        return f"❌ All HuggingFace models failed.\n\nErrors:\n{error_summary}\n\n**Suggestion:** Models may be loading. Wait 30-60 seconds and try again, or use OpenAI/xAI."

    def _call_model(self, model: str, prompt: str, temperature: float) -> str:
        url = f"https://api-inference.huggingface.co/models/{model}"
        
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
                        "temperature": max(temperature, 0.1),
                        "top_p": 0.9,
                        "return_full_text": False,
                        "do_sample": True
                    },
                    "options": {
                        "wait_for_model": True,
                        "use_cache": True
                    }
                },
                timeout=120.0
            )
            
            log_debug(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                log_debug(f"Response data type: {type(data)}")
                
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text", "").strip()
                    if text:
                        return text
                    return "❌ Empty response from model"
                elif isinstance(data, dict):
                    if "generated_text" in data:
                        return data["generated_text"].strip()
                    if "error" in data:
                        error_msg = data["error"]
                        if "loading" in error_msg.lower():
                            return "⏳ Model loading..."
                        return f"❌ {error_msg}"
                return f"❌ Unexpected response format: {str(data)[:100]}"
            
            elif response.status_code == 503:
                try:
                    data = response.json()
                    if "estimated_time" in data:
                        return f"⏳ Model loading ({int(data['estimated_time'])}s)..."
                except:
                    pass
                return "⏳ Model loading..."
            
            elif response.status_code == 401:
                return "❌ Invalid API key"
            
            elif response.status_code == 429:
                return "❌ Rate limited"
            
            elif response.status_code == 404:
                return "❌ Model not found"
            
            else:
                return f"❌ HTTP {response.status_code}: {response.text[:100]}"
                
        except httpx.TimeoutException:
            return "❌ Request timeout"
        except Exception as e:
            log_error(f"Exception calling {model}: {e}")
            return f"❌ Error: {str(e)[:50]}"

    def stream(self, messages: List[Dict], temperature: float = 0.3):
        yield self.generate(messages, temperature)

    def _format_messages(self, messages: List[Dict]) -> str:
        formatted = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                formatted += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}</s>\n"
        formatted += "<|assistant|>\n"
        return formatted


class OpenAILLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gpt-4o-mini"
        if api_key:
            log_info(f"OpenAILLM initialized with key: {api_key[:10]}...")
        else:
            log_error("OpenAILLM initialized WITHOUT key!")
    
    def generate(self, messages: List[Dict], temperature: float = 0.3) -> str:
        if not self.api_key:
            return "❌ No OpenAI API key provided"
            
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
                return f"❌ OpenAI Error {response.status_code}: {response.text[:200]}"
                
        except Exception as e:
            log_error(f"OpenAI exception: {e}")
            return f"❌ OpenAI Error: {e}"

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
            yield f"❌ Error: {e}"


class XaiLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "grok-beta"
        self.base_url = "https://api.x.ai/v1"
        if api_key:
            log_info(f"XaiLLM initialized with key: {api_key[:10]}...")
        else:
            log_error("XaiLLM initialized WITHOUT key!")

    def generate(self, messages: List[Dict], temperature: float = 0.3) -> str:
        if not self.api_key:
            return "❌ No xAI API key provided"
            
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
                return f"❌ xAI Error {response.status_code}: {error}"
                
        except Exception as e:
            log_error(f"xAI exception: {e}")
            return f"❌ xAI connection error: {e}"

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
                if response.status_code != 200:
                    yield f"❌ xAI Error {response.status_code}"
                    return
                    
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
            yield f"❌ Streaming error: {e}"


# ============================================================================
# HELPER: Get API Key with logging - SIMPLIFIED AND FIXED
# ============================================================================

def get_secret(key_name: str) -> Optional[str]:
    """
    Safely get a secret from Streamlit secrets or environment.
    Handles both flat and nested secret structures.
    """
    
    # Method 1: Direct access (flat structure)
    try:
        value = st.secrets[key_name]
        if value is not None:
            log_debug(f"✅ Found '{key_name}' directly in st.secrets")
            return str(value)
    except KeyError:
        pass
    except Exception as e:
        log_debug(f"Direct access to '{key_name}' failed: {e}")
    
    # Method 2: Search in nested sections
    try:
        for section_name in st.secrets:
            section = st.secrets[section_name]
            if isinstance(section, dict) or hasattr(section, 'keys'):
                try:
                    if key_name in section:
                        value = section[key_name]
                        if value is not None:
                            log_debug(f"✅ Found '{key_name}' in st.secrets['{section_name}']")
                            return str(value)
                except:
                    pass
    except Exception as e:
        log_debug(f"Nested search for '{key_name}' failed: {e}")
    
    # Method 3: Environment variable
    value = os.environ.get(key_name)
    if value:
        log_debug(f"✅ Found '{key_name}' in environment variables")
        return value
    
    log_debug(f"❌ Secret '{key_name}' not found anywhere")
    return None


def list_available_secrets() -> List[str]:
    """List all available secret keys for debugging"""
    secrets = []
    try:
        if hasattr(st, 'secrets'):
            for key in st.secrets:
                secrets.append(str(key))
    except Exception as e:
        log_warning(f"Error listing secrets: {e}")
    return secrets


def debug_secrets():
    """Debug function to check secrets loading"""
    info = {
        "has_st_secrets": hasattr(st, 'secrets'),
        "secrets_keys": [],
        "test_values": {}
    }
    
    try:
        if hasattr(st, 'secrets'):
            info["secrets_keys"] = list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else []
            
            # Test each expected key
            for key in ["HUGGINGFACE_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "MODEL_PASSWORD", "ADMIN_PASSWORD"]:
                try:
                    val = st.secrets.get(key, None) if hasattr(st.secrets, 'get') else st.secrets[key] if key in st.secrets else None
                    if val:
                        info["test_values"][key] = f"{str(val)[:8]}..." if len(str(val)) > 8 else "***"
                    else:
                        info["test_values"][key] = "NOT FOUND"
                except:
                    info["test_values"][key] = "ERROR"
    except Exception as e:
        info["error"] = str(e)
    
    return info


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
            ["openai", "xai", "huggingface"],
            format_func=lambda x: {
                "huggingface": "🤗 HuggingFace (Free - May be slow)",
                "openai": "🧠 GPT-4o-mini (OpenAI)",
                "xai": "🚀 Grok (xAI)"
            }[x]
        )
        
        log_debug(f"Selected model: {model_choice}")
        
        # Initialize variables
        api_key = None
        auth_valid = False
        
        # Get the MODEL_PASSWORD from secrets for comparison
        stored_password = get_secret("MODEL_PASSWORD")
        log_debug(f"Stored password found: {bool(stored_password)}")
        
        # KEY HANDLING LOGIC - FIXED
        if model_choice == "huggingface":
            # HuggingFace - try secrets first, then ask user
            api_key = get_secret("HUGGINGFACE_API_KEY")
            
            if api_key:
                st.success("✅ HuggingFace key loaded from secrets")
                auth_valid = True
            else:
                api_key = st.text_input(
                    "HuggingFace API Key:",
                    type="password",
                    help="Get a free key at huggingface.co/settings/tokens",
                    key="hf_key_input"
                )
                if api_key and len(api_key) > 10:
                    st.success("✅ Key provided")
                    auth_valid = True
                else:
                    st.warning("⚠️ Please enter your HuggingFace API key")
                    auth_valid = False
                    
        elif model_choice in ["openai", "xai"]:
            # OpenAI and xAI need password authentication
            pwd = st.text_input(
                "Access Password:", 
                type="password",
                key="model_password_input",
                help="Enter the password to access premium models"
            )
            
            if pwd:
                # Compare passwords
                if stored_password and pwd.strip() == stored_password.strip():
                    log_debug("Password matched!")
                    
                    # Get the appropriate API key
                    if model_choice == "openai":
                        api_key = get_secret("OPENAI_API_KEY")
                    else:
                        api_key = get_secret("XAI_API_KEY")
                    
                    if api_key:
                        st.success(f"✅ Authenticated - {model_choice.upper()} ready")
                        auth_valid = True
                        log_debug(f"API key loaded for {model_choice}: {api_key[:10]}...")
                    else:
                        st.error(f"❌ {model_choice.upper()}_API_KEY not found in secrets")
                        log_error(f"API key not found for {model_choice}")
                        auth_valid = False
                else:
                    st.error("❌ Invalid password")
                    log_debug(f"Password mismatch. Entered: '{pwd[:3]}...' Expected: '{stored_password[:3] if stored_password else 'None'}...'")
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

        # Debug Panel - ENHANCED
        with st.expander("🔧 Debug Panel"):
            st.write("**Secrets Debug:**")
            secrets_debug = debug_secrets()
            st.json(secrets_debug)
            
            st.write("**Current Auth State:**")
            st.write(f"- model_choice: `{model_choice}`")
            st.write(f"- auth_valid: `{auth_valid}`")
            st.write(f"- api_key exists: `{bool(api_key)}`")
            if api_key:
                st.write(f"- api_key preview: `{api_key[:10]}...`")
            
            st.write("**Recent Logs:**")
            if st.session_state.debug_logs:
                for log_entry in st.session_state.debug_logs[-15:]:
                    st.text(log_entry)
            else:
                st.info("No logs yet")
            
            if st.button("Clear Logs"):
                st.session_state.debug_logs = []
                st.rerun()

        # Admin Tools
        with st.expander("🔐 Admin Tools"):
            admin_pw = st.text_input("Admin Password:", type="password", key="admin_pw")
            expected_admin = get_secret("ADMIN_PASSWORD")
            
            if st.button("🔄 Refresh DB"):
                if expected_admin and admin_pw == expected_admin:
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
                    if model_choice == "huggingface":
                        error_msg = "⚠️ Please enter your HuggingFace API key in the sidebar."
                    else:
                        error_msg = f"⚠️ Please enter the correct password to use **{model_choice.upper()}**."
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
                    log_info(f"Creating {model_choice} LLM with key: {api_key[:10]}...")
                    
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
