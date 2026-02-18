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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse, urldefrag
import time

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
        logger.info(f"Added batch {i//batch_size + 1}/{total_batches}")
    
    logger.info(f"Vector store built with {collection.count()} chunks")
    return collection

# ============================================================================
# KNOWLEDGE BASE BUILDER
# ============================================================================

def build_knowledge_base(force_rebuild=False):
    """Build or load the knowledge base with enhanced error handling."""
    cache_path = Path(SCRAPE_CACHE_FILE)
    chroma_path = Path(CHROMA_DIR)
    
    # Check if we should attempt to load existing KB
    if not force_rebuild and chroma_path.exists() and cache_path.exists():
        try:
            logger.info("Attempting to load existing knowledge base...")
            metadata = json.loads(cache_path.read_text())
            
            # Verify the collection actually exists and is accessible
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            
            try:
                collection = client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=embedding_fn
                )
                count = collection.count()
                logger.info(f"Successfully loaded existing KB with {count} chunks")
                return True, f"Knowledge base loaded ({metadata.get('scraped_at', 'unknown')[:10]}, {count} chunks)", metadata
            except Exception as e:
                logger.warning(f"Collection exists but couldn't be loaded: {e}")
                logger.info("Will rebuild from scratch...")
                # Fall through to rebuild
        except Exception as e:
            logger.warning(f"Failed to load existing KB: {e}")
            # Fall through to rebuild
    
    # Need to rebuild
    logger.info("=" * 60)
    logger.info("BUILDING KNOWLEDGE BASE FROM SCRATCH")
    logger.info("=" * 60)
    
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    log_container = st.expander("📋 Detailed Build Log", expanded=False)
    
    def log_message(msg, level="info"):
        """Log to both logger and Streamlit UI."""
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
        # Clean up any corrupted data
        log_message("Cleaning up any existing data...")
        if chroma_path.exists():
            import shutil
            shutil.rmtree(chroma_path)
            log_message(f"Removed old chroma_db directory")
        if cache_path.exists():
            cache_path.unlink()
            log_message(f"Removed old metadata file")
        
        # Step 1: Crawl
        log_message("=" * 40)
        log_message("STEP 1: CRAWLING DOCUMENTATION")
        log_message("=" * 40)
        status_placeholder.info("📥 Step 1/3: Crawling D2L documentation...")
        
        crawler = DocCrawler()
        crawled_count = [0]  # Use list for closure mutability
        
        def crawl_progress(count, url):
            crawled_count[0] = count
            progress = min(count / MAX_PAGES, 0.33)
            progress_bar.progress(progress)
            status_placeholder.info(f"📥 Crawling page {count}/{MAX_PAGES}...")
            if count % 10 == 0:  # Log every 10th page
                log_message(f"Crawled {count} pages so far...")
        
        pages = crawler.crawl_all(max_pages=MAX_PAGES, progress_callback=crawl_progress)
        crawler.close()
        
        log_message(f"✓ Crawling complete: {len(pages)} pages retrieved")
        
        if not pages:
            raise Exception("No pages were crawled successfully. Check your internet connection.")
        
        # Step 2: Chunk
        log_message("=" * 40)
        log_message("STEP 2: CREATING CHUNKS")
        log_message("=" * 40)
        status_placeholder.info("✂️ Step 2/3: Creating searchable chunks...")
        
        processed_pages = [0]
        
        def chunk_progress(current, total):
            processed_pages[0] = current
            progress = 0.33 + (current / total) * 0.33
            progress_bar.progress(progress)
            if current % 20 == 0 or current == total:
                log_message(f"Processed {current}/{total} pages into chunks")
        
        chunks = create_chunks(pages, progress_callback=chunk_progress)
        
        log_message(f"✓ Chunking complete: {len(chunks)} chunks created")
        
        if not chunks:
            raise Exception("No chunks were created from the crawled pages.")
        
        # Verify chunk uniqueness
        chunk_ids = [c.chunk_id for c in chunks]
        unique_ids = set(chunk_ids)
        if len(chunk_ids) != len(unique_ids):
            duplicates = len(chunk_ids) - len(unique_ids)
            log_message(f"⚠ Warning: Found {duplicates} duplicate chunk IDs (should not happen)", "warning")
        else:
            log_message(f"✓ All {len(chunk_ids)} chunk IDs are unique")
        
        # Step 3: Embed
        log_message("=" * 40)
        log_message("STEP 3: BUILDING VECTOR STORE")
        log_message("=" * 40)
        status_placeholder.info("🧠 Step 3/3: Building vector embeddings (this takes a few minutes)...")
        
        log_message("Initializing ChromaDB...")
        embedded_batches = [0]
        
        def embed_progress(batch, total_batches):
            embedded_batches[0] = batch
            progress = 0.66 + (batch / total_batches) * 0.34
            progress_bar.progress(progress)
            status_placeholder.info(f"🧠 Embedding batch {batch}/{total_batches}...")
            log_message(f"Embedded batch {batch}/{total_batches}")
        
        collection = build_vector_store(chunks, progress_callback=embed_progress)
        
        final_count = collection.count()
        log_message(f"✓ Vector store complete: {final_count} vectors stored")
        
        # Save metadata
        metadata = {
            "scraped_at": datetime.utcnow().isoformat(),
            "pages_count": len(pages),
            "chunks_count": len(chunks),
            "vectors_count": final_count,
            "max_pages_crawled": MAX_PAGES,
        }
        cache_path.write_text(json.dumps(metadata, indent=2))
        log_message(f"✓ Metadata saved to {SCRAPE_CACHE_FILE}")
        
        # Final verification
        log_message("=" * 40)
        log_message("VERIFICATION")
        log_message("=" * 40)
        log_message(f"Pages crawled: {len(pages)}")
        log_message(f"Chunks created: {len(chunks)}")
        log_message(f"Vectors stored: {final_count}")
        log_message(f"Build completed at: {metadata['scraped_at']}")
        
        progress_bar.progress(1.0)
        status_placeholder.success(
            f"✅ Knowledge base built successfully!\n\n"
            f"📄 Pages: {len(pages)}\n"
            f"✂️ Chunks: {len(chunks)}\n"
            f"🧠 Vectors: {final_count}\n\n"
            f"You can now start asking questions!"
        )
        
        log_message("=" * 60)
        log_message("BUILD COMPLETE - READY TO USE")
        log_message("=" * 60)
        
        return True, "Knowledge base built successfully", metadata
        
    except Exception as e:
        error_msg = f"Failed to build knowledge base: {str(e)}"
        logger.error(error_msg, exc_info=True)
        log_message(f"❌ ERROR: {error_msg}", "error")
        status_placeholder.error(f"❌ {error_msg}\n\nCheck the build log for details.")
        return False, error_msg, {}

# ============================================================================
# STREAMLIT APP
# ============================================================================

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
                                import shutil
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
                                import shutil
                                if Path(CHROMA_DIR).exists():
                                    shutil.rmtree(CHROMA_DIR)
                                if Path(SCRAPE_CACHE_FILE).exists():
                                    Path(SCRAPE_CACHE_FILE).unlink()
                                st.success("✅ KB deleted. Refresh page to rebuild.")
                            else:
                                st.error("❌ Wrong password")
                        except Exception as e:
                            st.error(f"Error: {e}")
        
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
