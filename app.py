"""
Streamlit Chat UI for the Multi-Agent Product Recommender.
Powered by LangGraph and Playwright.
"""
import asyncio
import uuid
import pandas as pd
import streamlit as st

from orchestrator.graph import run_agent

# --- Page Config ---
st.set_page_config(
    page_title="Product Recommender — Multi-Agent AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Chat message styling */
    .chat-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }

    .chat-assistant {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        color: #e0e0e0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.95rem;
    }

    /* Agent badge */
    .agent-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.3);
        color: #a0b4ff;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-bottom: 10px;
        border: 1px solid rgba(102, 126, 234, 0.4);
    }

    /* Sidebar header */
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 16px;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        border-radius: 12px !important;
    }

    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of {"role": "user"/"assistant", "content": str, ...}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def send_message(message: str) -> dict:
    """Run the user message through the LangGraph Orchestrator."""
    try:
        # Build chat history for context
        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.chat_history[-10:]
        ]
        
        # Run the async graph in a synchronous Streamlit context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_agent(message, history))
        
        return result
        
    except Exception as e:
        return {
            "final_response": f"❌ Error: {str(e)}",
            "agent_used": None,
            "products": [],
        }


def render_product_table(products: list):
    """Renders a list of product dictionaries as a clean Pandas DataFrame."""
    if not products:
        return
        
    st.markdown("#### 📊 Cross-Platform Price Comparison")
    
    # Create DataFrame
    df = pd.DataFrame(products)
    
    # Configure columns for a beautiful layout
    column_config = {
        "store": st.column_config.TextColumn("Store"),
        "title": st.column_config.TextColumn("Product Name"),
        "price": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
        "currency": None, # Hide currency since we added ₹ to the price format
        "product_url": st.column_config.LinkColumn("Product Link", display_text="View Link 🔗"),
        "snippet": None, # Hide snippet if it exists in data
        "platform": None # Hide duplicate platform key if it exists
    }

    st.dataframe(
        df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
    )


# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">🛍️ Product Recommender</div>', unsafe_allow_html=True)
    st.markdown("---")

    # YouTube URL input
    st.markdown("### 📹 Ingest YouTube Video")
    youtube_url = st.text_input(
        "Paste YouTube URL:",
        placeholder="https://youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
    
    if st.button("🚀 Ingest Video", use_container_width=True) and youtube_url:
        with st.spinner("Ingesting video... This may take a few minutes."):
            result = send_message(youtube_url)
            
            # Add the user's URL to the chat
            st.session_state.chat_history.append({"role": "user", "content": f"📹 {youtube_url}"})
            
            # Try to grab the agent's actual text, or default to our custom welcome message
            bot_reply = result.get("message") or result.get("final_response")
            
            if not bot_reply or bot_reply == "Something went wrong.":
                bot_reply = "✅ **Video ingested successfully!** If you have any questions related to the product video, please ask."

            # Save the assistant's reply
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": bot_reply,
                "agent_used": result.get("agent_used", "Ingestion Agent"),
                "products": result.get("products", []),
            })
            st.rerun()

            
    st.markdown("---")
    st.markdown("### 💡 How to Use")
    st.markdown("""
    1. **Paste a YouTube URL** above to ingest a product video.
    2. **Ask questions** about the video in the chat.
    3. **Request recommendations** — ask to compare prices across Amazon, Flipkart, Myntra, and Croma!
    """)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# --- Main Chat Area ---
st.markdown("# 🛍️ Multi-Agent AI Recommender")
st.markdown("*Powered by A2A Protocol, LangGraph & Playwright*")
st.markdown("---")

# Display historical chat messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            agent = msg.get("agent_used")
            if agent:
                st.markdown(f'<span class="agent-badge">🤖 {agent}</span>', unsafe_allow_html=True)
            
            st.markdown(msg["content"])

            # Render data table if products exist
            products = msg.get("products", [])
            if products:
                render_product_table(products)

# Chat input block
if prompt := st.chat_input("Ask about products or compare prices..."):
    # Immediately render user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Process and render assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent is searching and comparing..."):
            result = send_message(prompt)

        agent = result.get("agent_used")
       # Look for 'message' (from RAG/Ingest) OR 'final_response' (from Search)
        content = result.get("message") or result.get("final_response", "Something went wrong.")
        products = result.get("products", [])

        if agent:
            st.markdown(f'<span class="agent-badge">🤖 {agent}</span>', unsafe_allow_html=True)
        
        st.markdown(content)
        
        if products:
            render_product_table(products)

        # Save to session state
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": content,
            "agent_used": agent,
            "products": products,
        })