"""
Orchestrator Router — Agent discovery and request routing.
Uses Qwen to classify user intent and delegate to the appropriate agent.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_ollama import ChatOllama
from a2a.client import A2AClient
from a2a.models import AgentCard, A2ATask
from config import AGENT_URLS, OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

INTENT_CLASSIFICATION_PROMPT = """You are a strict request router for a multi-agent system. Classify the user's message into one of these intents based strictly on the rules below.

AVAILABLE AGENTS:
{agent_descriptions}

USER MESSAGE: {message}
CONTEXT: {context}

Classify this message into exactly ONE of these intents:
- "ingest_video": ONLY use this if the user provides an actual YouTube link starting with "http://" or "https://". Do NOT use this if they ask for a summary.
- "chat_about_video": Use this if the user asks for a summary, asks a question about the video, or asks to analyze "it" (e.g., "Summarize the content of it", "What are the key points?").
- "search_products": Use this if the user wants to search for products online, compare prices, or check Amazon/Flipkart.
- "general": ONLY use this for basic greetings (e.g., "hi", "hello"). Do NOT use this for summaries or product questions.

Respond with ONLY the exact intent string, nothing else.

INTENT:"""


class OrchestratorRouter:
    """Discovers agents, classifies intents, and routes requests."""

    def __init__(self):
        self.a2a_client = A2AClient(timeout=180.0)
        self.agents: dict[str, AgentCard] = {}
        self.llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=0.1,  # Low temp for classification
        )

    async def discover_agents(self):
        """Discover all agents by fetching their Agent Cards."""
        for url in AGENT_URLS:
            try:
                card = await self.a2a_client.get_agent_card(url)
                self.agents[card.name] = card
                logger.info(f"Discovered agent: {card.name} at {card.url}")
            except Exception as e:
                logger.warning(f"Failed to discover agent at {url}: {e}")

        logger.info(f"Total agents discovered: {len(self.agents)}")
        return self.agents

    def _get_agent_descriptions(self) -> str:
        """Format agent descriptions for the intent classifier."""
        if not self.agents:
            return "No agents discovered yet."

        parts = []
        for name, card in self.agents.items():
            caps = ", ".join(card.capabilities)
            parts.append(f"- {name}: {card.description} (capabilities: {caps})")
        return "\n".join(parts)

    async def classify_intent(
        self, message: str, context: str = ""
    ) -> str:
        """Classify the user's intent to determine which agent to use."""
        # Quick URL check for YouTube links
        if any(domain in message.lower() for domain in ["youtube.com", "youtu.be"]):
            return "ingest_video"

        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            agent_descriptions=self._get_agent_descriptions(),
            message=message,
            context=context,
        )

        response = self.llm.invoke(prompt)
        intent = response.content.strip().lower().strip('"\'')

        # Validate the intent
        valid_intents = {"ingest_video", "chat_about_video", "search_products", "general"}
        if intent not in valid_intents:
            # Fallback classification based on keywords
            msg_lower = message.lower()
            if any(kw in msg_lower for kw in ["search", "find", "buy", "recommend", "price", "compare", "similar"]):
                intent = "search_products"
            elif any(kw in msg_lower for kw in ["video", "transcript", "what did", "tell me about"]):
                intent = "chat_about_video"
            else:
                intent = "general"

        logger.info(f"Classified intent: '{intent}' for message: '{message[:50]}...'")
        return intent

    def _get_agent_for_intent(self, intent: str) -> AgentCard | None:
        """Map an intent to the appropriate agent."""
        intent_to_agent = {
            "ingest_video": "YouTube RAG Agent",
            "chat_about_video": "YouTube RAG Agent",
            "search_products": "Product Search MCP Agent",
            "compare_prices": "Product Search MCP Agent",
        }

        agent_name = intent_to_agent.get(intent)
        if agent_name:
            return self.agents.get(agent_name)
        return None

    async def route_request(
        self,
        message: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict:
        """
        Route a user request to the appropriate agent.

        Returns:
            Dict with response message, agent used, and any product data.
        """
        if not self.agents:
            await self.discover_agents()

        if chat_history is None:
            chat_history = []

        # Classify intent
        context = " | ".join([m.get("content", "") for m in chat_history[-4:]])
        intent = await self.classify_intent(message, context)

        # Handle general queries locally
        if intent == "general":
            return await self._handle_general(message)

        # Get the right agent
        agent = self._get_agent_for_intent(intent)
        if not agent:
            return {
                "message": "I couldn't find the right agent to handle your request. Please try rephrasing.",
                "agent_used": None,
                "products": [],
            }

        # Build and send A2A task
        if intent == "ingest_video":
            task = A2ATask(
                capability="ingest_video",
                input_data={"youtube_url": message.strip()},
            )
        elif intent == "chat_about_video":
            task = A2ATask(
                capability="chat_about_video",
                input_data={
                    "query": message,
                    "chat_history": chat_history,
                },
            )
        elif intent == "search_products":
            task = A2ATask(
                capability="search_products",
                input_data={
                    "product_query": message,
                    "context": context,
                },
            )
        else:
            return await self._handle_general(message)

        # Send to agent
        try:
            logger.info(f"Sending task to {agent.name} at {agent.url}")
            response = await self.a2a_client.send_task(agent.url, task)

            if response.status == "success":
                return self._format_success_response(intent, response.result, agent.name)
            else:
                error_msg = response.error.get("message", "Unknown error") if response.error else "Unknown error"
                return {
                    "message": f"⚠️ {agent.name} encountered an error: {error_msg}",
                    "agent_used": agent.name,
                    "products": [],
                }
        except Exception as e:
            logger.error(f"Failed to communicate with {agent.name}: {e}", exc_info=True)
            return {
                "message": f"❌ Could not reach {agent.name}. Please make sure it's running on {agent.url}.",
                "agent_used": agent.name,
                "products": [],
            }

    def _format_success_response(self, intent: str, result: dict, agent_name: str) -> dict:
        """Format the agent's response for the Streamlit UI."""
        if intent == "ingest_video":
            status = result.get("status", "unknown")
            if status == "duplicate":
                msg = f"ℹ️ This video has already been ingested. You can ask questions about it!"
            else:
                products = result.get("products", [])
                product_list = "\n".join(
                    [f"  • **{p.get('name', 'Unknown')}** by {p.get('brand', 'Unknown')} ({p.get('category', '')})"
                     for p in products]
                )
                msg = (
                    f"✅ **Video ingested successfully!**\n\n"
                    f"📹 **Title**: {result.get('video_title', 'Unknown')}\n"
                    f"📝 **Transcript source**: {result.get('transcript_source', 'unknown')}\n"
                    f"📦 **Products found**: {result.get('products_found', 0)}\n"
                    f"📊 **Chunks stored**: {result.get('transcript_chunks_stored', 0)}\n"
                )
                if product_list:
                    msg += f"\n**Extracted Products:**\n{product_list}"

            return {
                "message": msg,
                "agent_used": agent_name,
                "products": result.get("products", []),
                "metadata": {"video_id": result.get("video_id")},
            }

        elif intent == "chat_about_video":
            return {
                "message": result.get("answer", "I couldn't find an answer."),
                "agent_used": agent_name,
                "products": [],
                "metadata": {"sources": result.get("sources", [])},
            }

        elif intent == "search_products":
            data = result.get("data", {})
            comparison = data.get("comparison_summary", "No comparison available.")
            platforms = data.get("platforms_searched", [])
            total = data.get("total_results", 0)

            msg = (
                f"🔍 **Product Search Results**\n\n"
                f"Searched across **{len(platforms)}** platforms: {', '.join(platforms)}\n"
                f"Found **{total}** results total.\n\n"
                f"---\n\n"
                f"{comparison}"
            )

            # Flatten results for product display
            flat_products = []
            for platform, items in data.get("results", {}).items():
                for item in items:
                    flat_products.append(item)

            return {
                "message": msg,
                "agent_used": agent_name,
                "products": flat_products,
                "metadata": {
                    "amazon_structured": data.get("amazon_structured", []),
                    "transaction_id": result.get("transaction_id"),
                },
            }

        return {
            "message": str(result),
            "agent_used": agent_name,
            "products": [],
        }

    async def _handle_general(self, message: str) -> dict:
        """Handle general/greeting messages locally."""
        response = self.llm.invoke(
            f"You are a helpful product recommendation assistant. The user can:\n"
            f"1. Paste a YouTube video URL to ingest product information\n"
            f"2. Ask questions about ingested video content\n"
            f"3. Search for similar products across e-commerce platforms\n\n"
            f"Respond helpfully to: {message}"
        )
        return {
            "message": response.content.strip(),
            "agent_used": "Orchestrator",
            "products": [],
        }
