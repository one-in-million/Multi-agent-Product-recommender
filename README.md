# 🛍️ Multi-Agent AI Product Recommender

An intelligent, multi-agent AI shopping assistant that combines **YouTube RAG (Retrieval-Augmented Generation)** with **Live E-commerce Web Scraping** to provide smart product recommendations and real-time price comparisons. 

Built with LangGraph, Streamlit, and local LLMs (Ollama), this application routes user queries to specialized AI agents to extract knowledge from video reviews and scrape live prices from platforms like Croma, Amazon, and Flipkart.

## ✨ Key Features

* **🧠 Multi-Agent Orchestration:** Uses LangGraph to intelligently route user prompts to the correct specialized agent based on intent (Video Q&A vs. Product Search).
* **📹 YouTube RAG Agent:** Ingests YouTube product review URLs, transcribes the audio, vectorizes it into ChromaDB, and allows users to ask highly specific questions about the video content.
* **🕵️ Live Price Scraper:** Uses a custom asynchronous `Playwright Stealth` scraper to bypass WAFs/Cloudflare and extract live pricing data from e-commerce giants.
* **📊 Cross-Platform Comparison:** Structures scraped data into clean Pandas DataFrames with clickable links for instant price comparisons.
* **🔒 Local & Private:** Powered by local LLMs via Ollama, meaning all AI processing runs completely locally with no API costs.

## 🏗️ Architecture & Tech Stack

* **Frontend UI:** Streamlit
* **Agent Orchestration:** LangGraph & LangChain
* **Local LLM Server:** Ollama (Qwen/Llama models)
* **Vector Database:** ChromaDB
* **Web Scraping:** Playwright Async, Playwright Stealth
* **Data Processing:** Pandas, Regex, Asyncio

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed and [Ollama](https://ollama.com/) running locally.

```bash
# Pull the required LLM model via Ollama
ollama run qwen2.5:7b  
```

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/multi-agent-product-recommender.git](https://github.com/yourusername/multi-agent-product-recommender.git)
cd multi-agent-product-recommender
```

### 3. Install Dependencies
Create a virtual environment and install the required Python packages:
```bash
python -m venv env

# On Windows:
env\Scripts\activate

# On macOS/Linux:
source env/bin/activate

pip install -r requirements.txt
```

### 4. Install Playwright Browsers
The scraper requires Chromium binaries to run the headless browser:
```bash
playwright install chromium
```

## 💻 Usage

Start the Streamlit application:
```bash
streamlit run app.py
```

### How to interact with the AI:
1.  **Ingest a Video:** Paste a YouTube URL into the sidebar and click "Ingest Video".
2.  **Ask Questions:** Type queries like *"What are the top 3 features of the phone in that video?"*
3.  **Compare Prices:** Ask the agent to search the web, e.g., *"Search online for the Samsung Galaxy S24 and compare prices."*

## 📂 Project Structure

```text
multi-agent-product-recommender/
│
├── app.py                              # Main Streamlit Chat UI
├── config.py                           # Environment variables and LLM config
│
├── orchestrator/
│   └── graph.py                        # LangGraph workflow and routing logic
│
├── agents/
│   ├── product_search/
│   │   └── mcp_search.py               # Search Agent integrating the scraper and LLM
│   └── youtube_rag/
│       └── ...                         # Video ingestion and ChromaDB logic
│
└── search/
    └── direct_scraper.py               # Playwright Stealth asynchronous e-commerce scraper
```

## ⚠️ Disclaimer
This project uses web scraping for educational and research purposes. E-commerce platforms frequently update their DOM structures and anti-bot protections (like CAPTCHAs and Cloudflare). Scraper functions may require periodic maintenance to adapt to these changes.
