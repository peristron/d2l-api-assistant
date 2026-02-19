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
# ChromaDB requires a newer version of SQLite than what is installed by default
# on Streamlit Cloud (Debian). We use pysqlite3-binary to override it.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If running locally without pysqlite3-binary, this passes silently
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
SCRAPE_CACHE_FILE = "scrape_metadata.json"
MAX_PAGES = 300
CRAWL_DELAY = 0.2

# Regex to capture API routes (e.g., GET /d2l/api/...)
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
        """Strip anchors and index.html to avoid duplicates."""
        url, _ = urldefrag(url)
        if url.endswith("/index.html"):
            url = url[:-10]
        return url.rstrip("/")

    def is_valid(self, url: str) -> bool:
        """Ensure we stay on the documentation domain and ignore assets."""
        parsed = urlparse(url)
        if parsed.hostname != "docs.valence.desire2learn.com":
            return False
        skip_ext = {".png", ".jpg", ".gif", ".css", ".js", ".zip", ".pdf", ".txt"}
        return not any(parsed.path.lower().endswith(ext) for ext in skip_ext)

    def crawl_page(self, url: str):
        try:
            response = self.client.get(url)
            # Handle 404s gracefully
            if response.status_code != 200:
                logger.warning(f"Skipping {url} (Status {response.status_code})")
                return None, []
            
            if "text/html" not in response.headers.get("content-type", ""):
                return None, []

            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            title = soup.find("title")
            title = title.get_text(strip=True) if title else "Untitled Page"
            
            # D2L docs usually have content in 'document' or 'rst-content' classes
            main = (
                soup.find("div", {"role": "main"}) or
                soup.find("div", class_="document") or
                soup.find("div", class_="rst-content") or
                soup.find("body")
            )
            
            if not main:
                return None, []
            
            content = main.get_text(separator="\n", strip=True)
            
            # Extract simple category from URL path (e.g. /lp/ or /le/)
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
            
            # Find all links
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
            logger.warning(f"Failed to crawl {url}: {e}")
            return None, []

    def crawl_all(self, max_pages=MAX_PAGES, progress_callback=None):
        queue = [self.normalize_url(BASE_URL)]
        self.visited.add(queue[0])
        
        while queue and len(self.pages) < max_pages:
            url = queue.pop(0)
            
            if progress_callback:
                progress_callback(len(self.pages) + 1, url)
            
            logger.info(f"Crawling [{len(self.pages)+1}]: {url}")
            
            page_data, links = self.crawl_page(url)
            
            if page_data and len(page_data["content"]) > 100: # Filter empty pages
                self.pages.append(page_data)
            
            # Add new links to queue
            for link in links:
                if link not in self.visited:
                    self.visited.add(link)
                    queue.append(link)
            
            time.sleep(CRAWL_DELAY)
        
        logger.info(f"Crawled {len(self.pages)} pages")
        return self.pages

    def close(self):
        self.client.close()

# ============================================================================
# CHUNKING & EMBEDDING
# ============================================================================

def create_chunks(pages, progress_callback=None):
    """Convert pages into searchable chunks."""
    chunks = []
    seen_ids = set()
    total_pages = len(pages)
    
    def make_unique_id(base_string):
        """Generate a unique ID, handling collisions."""
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
        
        # 1. Summary Chunk (The top of the page)
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
        
        # 2. Extract specific API routes
        seen_routes_on_page = set()
        for match in ROUTE_PATTERN.finditer(content):
            method = match.group(1).upper()
            path = match.group(2)
            
            route_key = f"{method}|{path}"
            if route_key in seen_routes_on_page:
                continue
            seen_routes_on_page.add(route_key)
            
            # Context window around the route
            start = max(0, match.start() - 300)
            end = min(len(content), match.end() + 1000)
            route_context = content[start:end]
            
            route_content = (
                f"# API Route: {method} {path}\n"
                f"Page: {title}\n"
                f"Source: {url}\n\n"
                f"{route_context}"
            )
            
            chunk_id = make_unique_id(f"{url}|route|{method}|{path}")
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
        
        # 3. Standard sliding window chunks
        words = content.split()
        chunk_size = 300
        overlap = 50
        
        part_num = 0
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text) < 50: continue
            
            full_content = (
                f"# {title}\n"
                f"Source: {url}\n"
                f"Part: {part_num + 1}\n\n"
                f"{chunk_text}"
            )
            
            chunk_id = make_unique_id(f"{url}|content|{part_num}")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=full_content,
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
    """Build ChromaDB vector store from chunks."""
    chroma_path = Path(CHROMA_DIR)
    
    # 1. Clean previous DB
    if chroma_path.exists():
        try:
            shutil.rmtree(chroma_path)
        except Exception as e:
            logger.warning(f"Could not delete folder immediately: {e}")
    
    chroma_path.mkdir(exist_ok=True)
    
    # 2. Initialize Client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 3. Create Collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    
    # 4. Add documents in batches
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
            # Chroma metadata must be primitives
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
    """Build or load the knowledge base."""
    cache_path = Path(SCRAPE_CACHE_FILE)
    chroma_path = Path(CHROMA_DIR)
    
    # Attempt Load
    if not force_rebuild and chroma_path.exists() and cache_path.exists():
        try:
            logger.info("Loading existing KB...")
            metadata = json.loads(cache_path.read_text())
            
            client = chromadb.PersistentClient(path=str(chroma_path))
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
            
            count = collection.count()
            if count > 0:
                return True, f"Loaded {count} chunks.", metadata
        except Exception as e:
            logger.warning(f"Load failed ({e}), rebuilding...")
    
    # Rebuild
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    log_container = st.expander("📋 Build Logs", expanded=True)
    
    def log(msg):
        with log_container:
            st.text(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        logger.info(msg)
    
    try:
        log("Starting clean build...")
        
        # Cleanup
        gc.collect()
        if chroma_path.exists():
            shutil.rmtree(chroma_path, ignore_errors=True)
        if cache_path.exists():
            cache_path.unlink()
        
        # Step 1: Crawl
        status_placeholder.info("📥 Step 1/3: Crawling Documentation...")
        crawler = DocCrawler()
        pages = crawler.crawl_all(
            max_pages=MAX_PAGES, 
            progress_callback=lambda c, u: progress_bar.progress(min(c / MAX_PAGES * 0.33, 0.33))
        )
        crawler.close()
        log(f"Crawled {len(pages)} pages.")
        
        if not pages:
            raise Exception("No pages found. Check connection.")

        # Step 2: Chunk
        status_placeholder.info("✂️ Step 2/3: Chunking content...")
        chunks = create_chunks(
            pages,
            progress_callback=lambda c, t: progress_bar.progress(0.33 + (c / t * 0.33))
        )
        log(f"Created {len(chunks)} chunks.")

        # Step 3: Embed
        status_placeholder.info("🧠 Step 3/3: Embedding vectors (this takes time)...")
        collection = build_vector_store(
            chunks,
            progress_callback=lambda b, t: progress_bar.progress(0.66 + (b / t * 0.34))
        )
        final_count = collection.count()
        log(f"Stored {final_count} vectors.")
        
        # Save Metadata
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
        log(f"ERROR: {e}")
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
            return chunks
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def build_messages(self, query, chunks, persona, history=None):
        system_prompt = DEVELOPER_PROMPT if persona == "developer" else PLAIN_PROMPT
        
        context_text = "\n\n".join([
            f"--- Source {i+1} ---\n{c['content']}" 
            for i, c in enumerate(chunks)
        ])
        
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-4:]) # Keep last 4 turns
        
        user_content = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer using the Context above."
        )
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
    """Free LLM via HuggingFace Router."""
    def __init__(self):
        self.model = "mistralai/Mistral-7B-Instruct-v0.3"
        # UPDATED: Use the Router URL
        self.base_url = "https://router.huggingface.co/hf"
    
    def generate(self, messages, temperature=0.3):
        prompt = self._format_messages(messages)
        try:
            response = httpx.post(
                f"{self.base_url}/models/{self.model}",
                headers={"Content-Type": "application/json"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 1500,
                        "temperature": temperature,
                        "return_full_text": False
                    }
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list):
                    return result[0].get("generated_text", "").strip()
                return str(result)
            elif response.status_code == 503:
                return "⏳ Model is loading on HuggingFace... try again in 30s."
            else:
                return f"Error {response.status_code}: {response.text[:100]}"
        except Exception as e:
            return f"Connection error: {e}"

    def stream(self, messages, temperature=0.3):
        # HF Inference API free tier doesn't support easy streaming
        # We simulate it by yielding the full result once
        full_response = self.generate(messages, temperature)
        yield full_response

    def _format_messages(self, messages):
        # Convert Chat format to Mistral instruction format
        out = ""
        for m in messages:
            if m["role"] == "user":
                out += f"[INST] {m['content']} [/INST]"
            elif m["role"] == "assistant":
                out += f" {m['content']} </s>"
            elif m["role"] == "system":
                out += f"[INST] <<SYS>>\n{m['content']}\n<</SYS>>\n"
        return out

class OpenAILLM:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o-mini"
    
    def generate(self, messages, temperature=0.3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "temperature": temperature},
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI Error: {e}"

    def stream(self, messages, temperature=0.3):
        try:
            with httpx.stream(
                "POST", "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "temperature": temperature, "stream": True, "max_tokens": 1500},
                timeout=60.0
            ) as response:
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            content = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                            if content: yield content
                        except: continue
        except Exception as e:
            yield f"Error: {e}"

class XaiLLM(OpenAILLM):
    """Grok (API compatible with OpenAI)"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "grok-beta" # or grok-3-mini
        self.base_url = "https://api.x.ai/v1"

    def generate(self, messages, temperature=0.3):
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "temperature": temperature},
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"xAI Error: {e}"

    def stream(self, messages, temperature=0.3):
        # Uses same stream logic as OpenAI, just different URL
        try:
            with httpx.stream(
                "POST", f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "temperature": temperature, "stream": True},
                timeout=60.0
            ) as response:
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            content = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                            if content: yield content
                        except: continue
        except Exception as e:
            yield f"Error: {e}"

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="D2L API Helper", page_icon="📚", layout="wide")
    st.title("📚 D2L Brightspace API Assistant")

    # Session State Init
    if "messages" not in st.session_state: st.session_state.messages = []
    if "kb_ready" not in st.session_state: st.session_state.kb_ready = False
    if "kb_metadata" not in st.session_state: st.session_state.kb_metadata = {}

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

    # Sidebar Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # KB Stats
        if st.session_state.kb_metadata:
             pages = st.session_state.kb_metadata.get('pages_count', '?')
             chunks = st.session_state.kb_metadata.get('chunks_count', '?')
             st.success(f"Loaded: {pages} pages, {chunks} chunks")

        persona = st.radio("Style:", ["developer", "plain"], format_func=lambda x: "👨‍💻 Developer" if x == "developer" else "📝 Plain English")
        st.divider()
        
        model_choice = st.selectbox("Model:", ["free", "openai", "xai"], format_func=lambda x: {"free": "Free (Mistral 7B)", "openai": "OpenAI (GPT-4o)", "xai": "xAI (Grok)"}[x])
        
        api_key = None
        if model_choice != "free":
            pwd = st.text_input("Access Password:", type="password")
            if pwd == st.secrets.get("MODEL_PASSWORD", "password"):
                api_key = st.secrets.get(f"{model_choice.upper()}_API_KEY")
                if api_key: st.success("Authenticated")
            else:
                st.warning("Enter password to use premium models")
                model_choice = "free"

        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        with st.expander("Admin Tools"):
            admin_pw = st.text_input("Admin PW:", type="password")
            if st.button("🔄 Refresh DB") and admin_pw == st.secrets.get("ADMIN_PASSWORD", "admin"):
                st.session_state.kb_ready = False
                del st.session_state.rag
                build_knowledge_base(force_rebuild=True)
                st.rerun()
            
            st.divider()
            
            # Diagnostic: Show what's in the DB
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
                    st.write(f"**Total Pages:** {len(urls)}")
                    st.write("**Categories:**")
                    st.json(categories)
                except Exception as e:
                    st.error(f"Error: {e}")
            
            # Diagnostic: Scan for missing pages
            if st.button("🔎 Scan Missing Pages"):
                with st.spinner("Scanning..."):
                    try:
                        scan_client = httpx.Client(timeout=10)
                        resp = scan_client.get(BASE_URL)
                        soup = BeautifulSoup(resp.text, "html.parser")
                        site_urls = set()
                        for a in soup.find_all("a", href=True):
                             href = a["href"]
                             if not href.startswith("#") and "mailto" not in href:
                                 full = urljoin(BASE_URL, href)
                                 if "docs.valence.desire2learn.com" in full:
                                     site_urls.add(full.split("#")[0])
                        
                        client = chromadb.PersistentClient(path=CHROMA_DIR)
                        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
                        db_data = collection.get(include=["metadatas"])
                        db_urls = set(m.get("source_url", "") for m in db_data["metadatas"])
                        
                        missing = [u for u in site_urls if u not in db_urls and u.endswith(".html")]
                        
                        if missing:
                            st.warning(f"Found {len(missing)} missing pages.")
                            with st.expander("Show Missing"):
                                for m in missing[:50]: st.write(m)
                        else:
                            st.success("All linked pages appear to be indexed.")
                    except Exception as e:
                        st.error(f"Scan failed: {e}")

    # Chat Interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(f"- [{s['title']}]({s['url']})")

    if prompt := st.chat_input("How do I get a user's enrolled courses?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching docs..."):
                # Retrieve
                chunks = st.session_state.rag.retrieve(prompt)
                sources = st.session_state.rag.get_sources(chunks)
                
                # Prepare LLM
                if model_choice == "openai" and api_key: llm = OpenAILLM(api_key)
                elif model_choice == "xai" and api_key: llm = XaiLLM(api_key)
                else: llm = HuggingFaceLLM()
                
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                messages = st.session_state.rag.build_messages(prompt, chunks, persona, history)
                
                # Stream Response
                response_placeholder = st.empty()
                full_response = ""
                
                stream = llm.stream(messages)
                for chunk in stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })

if __name__ == "__main__":
    main()
