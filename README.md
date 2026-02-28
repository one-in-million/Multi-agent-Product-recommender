# 🛍️ Multi-Agent Product Recommender

A federated multi-agent system using a simplified A2A (Agent-to-Agent) protocol. Ingests YouTube product videos, extracts product information, and recommends similar products across e-commerce platforms.

## Architecture

```
┌──────────────┐     HTTP      ┌──────────────────┐     A2A      ┌─────────────────────┐
│  Streamlit   │◄────────────►│   Orchestrator    │◄──────────►│  Agent 1: YouTube   │
│  Chat UI     │              │   (port 8000)     │            │  RAG (port 8001)    │
└──────────────┘              │                    │            └─────────────────────┘
                              │  - Intent classify │
                              │  - Agent discovery │     A2A      ┌─────────────────────┐
                              │  - Response agg.   │◄──────────►│  Agent 2: Product   │
                              └──────────────────┘            │  Search (port 8002)  │
                                                              └─────────────────────┘
```

## Tech Stack

| Component | Tech |
|-----------|------|
| LLM | Qwen2.5:7b (Ollama) |
| Speech-to-Text | Whisper base (CPU) |
| Subtitles | youtube-transcript-api |
| Vector DB | PGVector (PostgreSQL) |
| Embeddings | all-MiniLM-L6-v2 |
| Product Search | DuckDuckGo + BeautifulSoup |
| Agent Servers | FastAPI + Uvicorn |
| UI | Streamlit |

## Prerequisites

1. **Python 3.10+**
2. **PostgreSQL** with PGVector extension
3. **Ollama** with `qwen2.5:7b` model
4. **ffmpeg** (for Whisper audio processing)

## Setup

### 1. Database Setup

```sql
-- Connect to PostgreSQL and run:
CREATE DATABASE product_recommender;
\c product_recommender
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Install Dependencies

```powershell
cd d:\Antigravity\Projects\product-recommender
pip install -r requirements.txt
```

### 3. Configure

Edit `config.py` to set your PostgreSQL credentials:
```python
PG_USER = "postgres"
PG_PASSWORD = "your_password"
PG_DATABASE = "product_recommender"
```

### 4. Verify Ollama

```powershell
ollama list
# Should show qwen2.5:7b
```

## Running

Open **4 separate terminals** and run from the project root:

```powershell
# Terminal 1: Agent 1 (YouTube RAG)
python -m agents.youtube_rag.server

# Terminal 2: Agent 2 (Product Search)
python -m agents.product_search.server

# Terminal 3: Orchestrator
python -m orchestrator.server

# Terminal 4: Streamlit UI
streamlit run app.py
```

## Usage

1. **Ingest a Video**: Paste a YouTube URL in the sidebar and click "Ingest Video"
2. **Ask Questions**: Chat about the video content (e.g., "What products were discussed?")
3. **Get Recommendations**: Ask for similar products (e.g., "Find similar products with prices")

## Project Structure

```
product-recommender/
├── app.py                           # Streamlit UI
├── config.py                        # Configuration
├── requirements.txt
├── a2a/                             # A2A Protocol
│   ├── models.py                    # Data models
│   └── client.py                    # HTTP client
├── orchestrator/                    # Central Orchestrator
│   ├── server.py                    # FastAPI (port 8000)
│   └── router.py                    # Intent routing
├── agents/
│   ├── youtube_rag/                 # Agent 1
│   │   ├── server.py                # FastAPI (port 8001)
│   │   ├── agent_card.json
│   │   ├── transcript_manager.py
│   │   └── rag_chain.py
│   └── product_search/              # Agent 2
│       ├── server.py                # FastAPI (port 8002)
│       ├── agent_card.json
│       ├── collector.py
│       └── mcp_search.py
├── core/                            # Shared modules
│   ├── youtube_downloader.py
│   ├── transcriber.py
│   ├── product_extractor.py
│   └── vector_store.py
├── search/
│   ├── web_search.py
│   └── amazon_scraper.py
└── utils/
    └── helpers.py
```

## License

MIT
