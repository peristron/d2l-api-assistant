"""
D2L Brightspace API Documentation Assistant
Single-file proof of concept with IN-APP scraping, embedding, and chat.

First run: Automatically scrapes and builds knowledge base (takes ~10 mins)
Subsequent runs: Loads from cache
Admin: Password-protected "Refresh Docs" button to re-scrape
"""

import streamlit as st
import json
import re
import hashlib
import logging
import shutil
import gc  # <-- Add this line
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin, urlparse, urldefrag
import time

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

    def crawl_all(self, max_pages=MAX_PAGES, progress_callback=None):
        queue = [self.normalize_url(BASE_URL)]
        
        while queue and len(self.pages) < max_pages:
            url = queue.pop(0)
            
            if url in self.visited:
                continue
            
            self.visited.add(url)
            
            if progress_callback:
                progress_callback(len(self.pages) + 1, url)
            
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

def create_chunks(pages, progress_callback=None):
    """Convert pages into searchable chunks."""
    chunks = []
    seen_ids = set()  # Track IDs we've already used
    total_pages = len(pages)
    
    def make_unique_id(base_string):
        """Generate a unique ID, handling collisions."""
        chunk_id = hashlib.md5(base_string.encode()).hexdigest()[:16]
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            return chunk_id
        # Handle collision by adding a counter
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
        
        # Create page summary chunk
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
        
        # Extract API routes - track which we've seen on this page
        seen_routes_on_page = set()
        for match in ROUTE_PATTERN.finditer(content):
            method = match.group(1).upper()
            path = match.group(2)
            
            # Skip if we've already captured this route on this page
            route_key = f"{method}|{path}"
            if route_key in seen_routes_on_page:
                continue
            seen_routes_on_page.add(route_key)
            
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
        
        # Split long content into overlapping chunks
        words = content.split()
        chunk_size = 300  # words
        overlap = 50
        
        part_num = 0
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
    
    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

def build_vector_store(chunks, progress_callback=None):
    """Build ChromaDB vector store from chunks."""
    import shutil
    
    chroma_path = Path(CHROMA_DIR)
    
    # Ensure completely clean slate
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        logger.info(f"Removed existing {CHROMA_DIR}")
    
    chroma_path.mkdir(exist_ok=True)
    
    # Create fresh client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # List and delete any existing collections (belt and suspenders)
    existing_collections = client.list_collections()
    for col in existing_collections:
        logger.info(f"Deleting existing collection: {col.name}")
        client.delete_collection(col.name)
    
    # Now create fresh collection
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    
    logger.info(f"Created new collection: {COLLECTION_NAME}")
    
    # Add chunks in batches
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
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)
        
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    
    logger.info(f"Vector store built with {collection.count()} chunks")
    return collection

# ============================================================================
# KNOWLEDGE BASE BUILDER
# ============================================================================

def build_knowledge_base(force_rebuild=False):
    """Build or load the knowledge base with enhanced error handling."""
    cache_path = Path(SCRAPE_CACHE_FILE)
    chroma_path = Path(CHROMA_DIR)
    
    # Check if we should load existing KB
    if not force_rebuild and chroma_path.exists() and cache_path.exists():
        try:
            logger.info("Attempting to load existing knowledge base...")
            metadata = json.loads(cache_path.read_text())
            
            # Verify the collection is accessible
            client = chromadb.PersistentClient(path=str(chroma_path))
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            
            collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_fn
            )
            count = collection.count()
            logger.info(f"Successfully loaded existing KB with {count} chunks")
            return True, f"Knowledge base loaded ({count} chunks)", metadata
            
        except Exception as e:
            logger.warning(f"Failed to load existing KB: {e}")
            logger.info("Will rebuild from scratch...")
    
    # Rebuild
    logger.info("=" * 60)
    logger.info("BUILDING KNOWLEDGE BASE FROM SCRATCH")
    logger.info("=" * 60)
    
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    log_container = st.expander("📋 Detailed Build Log", expanded=True)
    
    def log_message(msg, level="info"):
        if level == "info":
            logger.info(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        with log_container:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.text(f"[{timestamp}] {msg}")
    
    try:
        # ============================================================
        # AGGRESSIVE CLEANUP - ensures completely fresh start
        # ============================================================
        log_message("Cleaning up any existing data...")
        
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            log_message(f"Removed old {CHROMA_DIR} directory")
            # Give filesystem time to complete deletion
            time.sleep(0.5)
        
        if cache_path.exists():
            cache_path.unlink()
            log_message(f"Removed old {SCRAPE_CACHE_FILE} file")
        
        # Force garbage collection to release any held references
        gc.collect()
        
        # Additional pause to ensure everything is cleaned up
        time.sleep(0.5)
        
        log_message("Cleanup complete, starting fresh build...")
        
        # ============================================================
        # STEP 1: CRAWL
        # ============================================================
        log_message("=" * 40)
        log_message("STEP 1: CRAWLING DOCUMENTATION")
        log_message("=" * 40)
        status_placeholder.info("📥 Step 1/3: Crawling D2L documentation...")
        
        crawler = DocCrawler()
        
        def crawl_progress(count, url):
            progress_bar.progress(min(count / MAX_PAGES, 0.33))
            status_placeholder.info(f"📥 Crawling page {count}/{MAX_PAGES}...")
            if count % 10 == 0:
                log_message(f"Crawled {count} pages so far...")
        
        pages = crawler.crawl_all(max_pages=MAX_PAGES, progress_callback=crawl_progress)
        crawler.close()
        
        log_message(f"✓ Crawling complete: {len(pages)} pages retrieved")
        
        if not pages:
            raise Exception("No pages crawled. Check internet connection.")
        
        # ============================================================
        # STEP 2: CHUNK
        # ============================================================
        log_message("=" * 40)
        log_message("STEP 2: CREATING CHUNKS")
        log_message("=" * 40)
        status_placeholder.info("✂️ Step 2/3: Creating searchable chunks...")
        
        def chunk_progress(current, total):
            progress_bar.progress(0.33 + (current / total) * 0.33)
            if current % 20 == 0 or current == total:
                log_message(f"Processed {current}/{total} pages")
        
        chunks = create_chunks(pages, progress_callback=chunk_progress)
        
        log_message(f"✓ Chunking complete: {len(chunks)} chunks created")
        
        if not chunks:
            raise Exception("No chunks created from pages.")
        
        # Verify uniqueness
        chunk_ids = [c.chunk_id for c in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise Exception("Duplicate chunk IDs detected!")
        log_message(f"✓ All {len(chunk_ids)} chunk IDs are unique")
        
        # ============================================================
        # STEP 3: EMBED
        # ============================================================
        log_message("=" * 40)
        log_message("STEP 3: BUILDING VECTOR STORE")
        log_message("=" * 40)
        status_placeholder.info("🧠 Step 3/3: Building vector embeddings...")
        
        log_message("Initializing ChromaDB...")
        
        def embed_progress(batch, total_batches):
            progress_bar.progress(0.66 + (batch / total_batches) * 0.34)
            status_placeholder.info(f"🧠 Embedding batch {batch}/{total_batches}...")
            log_message(f"Embedded batch {batch}/{total_batches}")
        
        collection = build_vector_store(chunks, progress_callback=embed_progress)
        
        final_count = collection.count()
        log_message(f"✓ Vector store complete: {final_count} vectors")
        
        # ============================================================
        # SAVE METADATA
        # ============================================================
        metadata = {
            "scraped_at": datetime.utcnow().isoformat(),
            "pages_count": len(pages),
            "chunks_count": len(chunks),
            "vectors_count": final_count,
        }
        cache_path.write_text(json.dumps(metadata, indent=2))
        log_message(f"✓ Metadata saved")
        
        # ============================================================
        # VERIFICATION & COMPLETE
        # ============================================================
        log_message("=" * 40)
        log_message("BUILD COMPLETE")
        log_message(f"  Pages: {len(pages)}")
        log_message(f"  Chunks: {len(chunks)}")
        log_message(f"  Vectors: {final_count}")
        log_message("=" * 40)
        
        progress_bar.progress(1.0)
        status_placeholder.success(
            f"✅ Knowledge base built!\n\n"
            f"📄 {len(pages)} pages | ✂️ {len(chunks)} chunks | 🧠 {final_count} vectors"
        )
        
        return True, "Success", metadata
        
    except Exception as e:
        error_msg = f"Failed to build knowledge base: {str(e)}"
        logger.error(error_msg, exc_info=True)
        log_message(f"❌ ERROR: {error_msg}", "error")
        status_placeholder.error(error_msg)
        return False, error_msg, {}

# ============================================================================
# STREAMLIT APP
# ============================================================================

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
            f"--- Source {i+1} [{c['relevance']:.0%} match] ---\n"
            f"URL: {c['metadata'].get('source_url', 'unknown')}\n"
            f"{c['content']}"
            for i, c in enumerate(chunks)
        ])
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            messages.extend(history[-6:])
        
        user_msg = (
            f"Documentation:\n\n{context}\n\n---\n\n"
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
# LLM PROVIDERS
# ============================================================================

class HuggingFaceLLM:
    """Free LLM via HuggingFace Inference API."""
    def __init__(self):
        self.model = "mistralai/Mistral-7B-Instruct-v0.3"
        self.base_url = "https://router.huggingface.co/hf"  # Updated endpoint
    
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
                timeout=120.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "").strip()
                return str(result)
            elif response.status_code == 503:
                return "⏳ Model is loading. Please wait 20-30 seconds and try again."
            else:
                return f"API error ({response.status_code}): {response.text[:200]}"
        except Exception as e:
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
            return f"OpenAI error: {e}"
    
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
            return f"xAI error: {e}"
    
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


def main():
    st.set_page_config(
        page_title="D2L API Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 D2L Brightspace API Assistant")
    
    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_ready" not in st.session_state:
        st.session_state.kb_ready = False
    if "kb_metadata" not in st.session_state:
        st.session_state.kb_metadata = {}
    
    # Build/load knowledge base
    if not st.session_state.kb_ready:
        st.info("🔍 Initializing knowledge base...")
        success, message, metadata = build_knowledge_base(force_rebuild=False)
        st.session_state.kb_ready = success
        st.session_state.kb_metadata = metadata
        
        if not success:
            st.error(f"**Failed to initialize knowledge base:**\n\n{message}")
            
            # Offer recovery options
            st.warning("### Recovery Options:")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Try Rebuilding", type="primary"):
                    # Force clean rebuild
                    import shutil
                    if Path(CHROMA_DIR).exists():
                        shutil.rmtree(CHROMA_DIR)
                    if Path(SCRAPE_CACHE_FILE).exists():
                        Path(SCRAPE_CACHE_FILE).unlink()
                    st.session_state.kb_ready = False
                    st.rerun()
            
            with col2:
                if st.button("📋 View Logs"):
                    st.info("Check the build log above for detailed error information.")
            
            st.stop()
        else:
            # Successfully loaded/built - force refresh to show chat interface
            time.sleep(1)  # Brief pause to show success message
            st.rerun()
    
    # Load RAG engine
    if "rag" not in st.session_state:
        try:
            with st.spinner("Loading search engine..."):
                st.session_state.rag = RAGEngine()
                logger.info("RAG engine loaded successfully")
        except Exception as e:
            st.error(f"Failed to load RAG engine: {e}")
            logger.error(f"RAG engine error: {e}", exc_info=True)
            
            if st.button("🔄 Reset and Rebuild"):
                import shutil
                if Path(CHROMA_DIR).exists():
                    shutil.rmtree(CHROMA_DIR)
                if Path(SCRAPE_CACHE_FILE).exists():
                    Path(SCRAPE_CACHE_FILE).unlink()
                st.session_state.kb_ready = False
                if "rag" in st.session_state:
                    del st.session_state.rag
                st.rerun()
            
            st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # KB info
        if st.session_state.kb_metadata:
            scraped_at = st.session_state.kb_metadata.get('scraped_at', 'unknown')
            if scraped_at != 'unknown':
                scraped_at = scraped_at[:10]
            pages = st.session_state.kb_metadata.get('pages_count', '?')
            chunks = st.session_state.kb_metadata.get('chunks_count', '?')
            vectors = st.session_state.kb_metadata.get('vectors_count', chunks)
            
            st.success(
                f"📊 **Knowledge Base**\n\n"
                f"📄 {pages} pages\n"
                f"✂️ {chunks} chunks\n"
                f"🧠 {vectors} vectors\n\n"
                f"📅 Updated: {scraped_at}"
            )
            
            with st.expander("🔧 Admin Tools"):
                admin_pw = st.text_input("Admin Password:", type="password", key="admin_pw")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Refresh Docs", help="Re-scrape documentation from D2L"):
                        try:
                            correct_pw = st.secrets.get("ADMIN_PASSWORD", "admin")
                            if admin_pw == correct_pw:
                                st.session_state.kb_ready = False
                                if "rag" in st.session_state:
                                    del st.session_state.rag
                                # Clean slate
                                if Path(CHROMA_DIR).exists():
                                    shutil.rmtree(CHROMA_DIR)
                                if Path(SCRAPE_CACHE_FILE).exists():
                                    Path(SCRAPE_CACHE_FILE).unlink()
                                st.rerun()
                            else:
                                st.error("❌ Wrong password")
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                with col2:
                    if st.button("🗑️ Reset KB", help="Delete and rebuild knowledge base"):
                        try:
                            correct_pw = st.secrets.get("ADMIN_PASSWORD", "admin")
                            if admin_pw == correct_pw:
                                if Path(CHROMA_DIR).exists():
                                    shutil.rmtree(CHROMA_DIR)
                                if Path(SCRAPE_CACHE_FILE).exists():
                                    Path(SCRAPE_CACHE_FILE).unlink()
                                st.success("✅ KB deleted. Refresh page to rebuild.")
                            else:
                                st.error("❌ Wrong password")
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                # Diagnostic buttons - no password required (read-only)
                st.divider()
                
                if st.button("🔍 Diagnose Crawl", help="Show what was indexed"):
                    try:
                        client = chromadb.PersistentClient(path=CHROMA_DIR)
                        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
                        
                        # Get all metadata
                        all_data = collection.get(include=["metadatas"])
                        
                        # Extract unique URLs and categories
                        urls = set()
                        categories = {}
                        chunk_types = {}
                        
                        for meta in all_data["metadatas"]:
                            url = meta.get("source_url", "")
                            if url:
                                urls.add(url)
                            
                            cat = meta.get("category", "unknown")
                            categories[cat] = categories.get(cat, 0) + 1
                            
                            ct = meta.get("chunk_type", "unknown")
                            chunk_types[ct] = chunk_types.get(ct, 0) + 1
                        
                        st.success(f"**Unique pages indexed:** {len(urls)}")
                        
                        st.write("**Chunk types:**")
                        for ct, count in sorted(chunk_types.items(), key=lambda x: -x[1]):
                            st.write(f"- {ct}: {count}")
                        
                        st.write("**Top 15 categories:**")
                        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:15]:
                            st.write(f"- `{cat}`: {count} chunks")
                        
                        with st.expander("📄 All indexed URLs"):
                            for url in sorted(urls):
                                st.write(f"- {url}")
                                
                    except Exception as e:
                        st.error(f"Diagnostic error: {e}")
                
                if st.button("🔎 Find Missing Pages", help="Scan for pages not yet indexed"):
                    with st.spinner("Scanning documentation site..."):
                        try:
                            check_urls = [
                                "https://docs.valence.desire2learn.com/",
                                "https://docs.valence.desire2learn.com/reference.html",
                                "https://docs.valence.desire2learn.com/http-routingtable.html",
                            ]
                            
                            discovered_urls = set()
                            scan_client = httpx.Client(timeout=30.0, follow_redirects=True)
                            
                            for check_url in check_urls:
                                try:
                                    response = scan_client.get(check_url)
                                    if response.status_code == 200:
                                        soup = BeautifulSoup(response.text, "html.parser")
                                        for a in soup.find_all("a", href=True):
                                            href = a["href"]
                                            if href.startswith("#") or href.startswith("mailto:"):
                                                continue
                                            abs_url = urljoin(check_url, href)
                                            parsed = urlparse(abs_url)
                                            if parsed.hostname == "docs.valence.desire2learn.com":
                                                clean = f"{parsed.scheme}://{parsed.hostname}{parsed.path}"
                                                clean = clean.rstrip("/")
                                                discovered_urls.add(clean)
                                except:
                                    pass
                            
                            scan_client.close()
                            
                            # Get indexed URLs
                            db_client = chromadb.PersistentClient(path=CHROMA_DIR)
                            embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                            collection = db_client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
                            all_data = collection.get(include=["metadatas"])
                            
                            indexed_urls = set()
                            for meta in all_data["metadatas"]:
                                url = meta.get("source_url", "")
                                if url:
                                    indexed_urls.add(url.rstrip("/"))
                            
                            # Filter to HTML pages only
                            html_pages = {u for u in discovered_urls 
                                         if u.endswith(".html") or not "." in u.split("/")[-1]}
                            
                            # Exclude utility pages
                            skip_patterns = ["search.html", "genindex.html", "py-modindex.html", "_sources", "_static"]
                            html_pages = {u for u in html_pages 
                                         if not any(skip in u for skip in skip_patterns)}
                            
                            # Find missing
                            missing = html_pages - indexed_urls
                            
                            st.info(f"**Found {len(html_pages)} content pages on site**")
                            st.info(f"**Currently indexed: {len(indexed_urls)} pages**")
                            
                            if missing:
                                st.warning(f"**Potentially missing: {len(missing)} pages**")
                                with st.expander("📋 Missing URLs"):
                                    for url in sorted(missing):
                                        st.write(f"- {url}")
                            else:
                                st.success("✅ All discoverable pages are indexed!")
                                
                        except Exception as e:
                            st.error(f"Error scanning: {e}")
        
        st.divider()
        
        persona = st.radio(
            "Response Style:",
            ["developer", "plain_english"],
            format_func=lambda x: "👨‍💻 Developer Mode" if x == "developer" else "📝 Plain English"
        )
        
        st.divider()
        
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
            password = st.text_input("Model Password:", type="password", key="model_pw")
            if password:
                try:
                    correct = st.secrets.get("MODEL_PASSWORD", "")
                    if password == correct:
                        st.success("✅ Access granted")
                        api_key = st.secrets.get(
                            "OPENAI_API_KEY" if model == "openai" else "XAI_API_KEY"
                        )
                    else:
                        st.error("❌ Wrong password")
                        model = "free"
                except Exception as e:
                    st.warning(f"Secrets error: {e}")
                    model = "free"
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Debug info
        with st.expander("🐛 Debug Info"):
            st.text(f"KB Ready: {st.session_state.kb_ready}")
            st.text(f"RAG Loaded: {'rag' in st.session_state}")
            st.text(f"Messages: {len(st.session_state.messages)}")
            if Path(CHROMA_DIR).exists():
                st.text(f"ChromaDB exists: ✓")
            else:
                st.text(f"ChromaDB exists: ✗")
            if Path(SCRAPE_CACHE_FILE).exists():
                st.text(f"Cache exists: ✓")
            else:
                st.text(f"Cache exists: ✗")
    
    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📖 Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        title_preview = src['title'][:60] + "..." if len(src['title']) > 60 else src['title']
                        st.markdown(f"- [{title_preview}]({src['url']}) ({src['relevance']:.0%})")
    
    # Welcome message
    if not st.session_state.messages:
        st.info(
            "👋 **Welcome!** I can answer questions about the D2L Brightspace API.\n\n"
            "Try asking:\n"
            "- *What are the authentication methods?*\n"
            "- *How do I enroll a user in a course?*\n"
            "- *Show me the parameters for GET /d2l/api/lp/{version}/users/*"
        )
    
    # Chat input
    if prompt := st.chat_input("Ask about the D2L API..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get LLM
        try:
            if model == "free":
                llm = HuggingFaceLLM()
            elif model == "openai":
                if not api_key:
                    raise ValueError("OpenAI API key not available")
                llm = OpenAILLM(api_key)
            else:
                if not api_key:
                    raise ValueError("xAI API key not available")
                llm = XaiLLM(api_key)
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Failed to initialize LLM: {e}")
            st.stop()
        
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documentation..."):
                    if model in ["openai", "xai"] and api_key:
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
                            title_preview = src['title'][:60] + "..." if len(src['title']) > 60 else src['title']
                            st.markdown(f"- [{title_preview}]({src['url']}) ({src['relevance']:.0%})")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Error generating response: {e}")
                logger.error(f"Query error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
