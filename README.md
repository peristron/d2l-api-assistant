TBD, but basically..

# D2L API Assistant - Simple / proof-of-concept Version

Single-file proof of concept for a RAG-powered Q&A assistant for D2L Brightspace API docs.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Scrape documentation and build vector store (takes ~10 minutes)
python app.py --scrape

# Run the app
streamlit run app.py
