"""
A2A Client — HTTP client for inter-agent communication.
Used by the Orchestrator to discover agents and send tasks.
"""
import httpx
from a2a.models import AgentCard, A2ATask, A2AResponse


class A2AClient:
    """HTTP client for communicating with A2A-compatible agents."""

    def __init__(self, timeout: float =400.0):
        self.timeout = timeout

    async def get_agent_card(self, agent_url: str) -> AgentCard:
        """
        Discover an agent by fetching its Agent Card.

        Args:
            agent_url: Base URL of the agent (e.g., http://localhost:8001)

        Returns:
            AgentCard with the agent's metadata and capabilities.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{agent_url}/agent-card")
            response.raise_for_status()
            return AgentCard(**response.json())

    async def send_task(self, agent_url: str, task: A2ATask) -> A2AResponse:
        """
        Send a task to an agent for processing.

        Args:
            agent_url: Base URL of the agent.
            task: The A2ATask to send.

        Returns:
            A2AResponse with the result.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{agent_url}/tasks",
                json=task.model_dump(),
            )
            response.raise_for_status()
            return A2AResponse(**response.json())

    async def get_task_status(self, agent_url: str, task_id: str) -> A2AResponse:
        """
        Check the status of a previously sent task.

        Args:
            agent_url: Base URL of the agent.
            task_id: ID of the task to check.

        Returns:
            A2AResponse with current status.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{agent_url}/tasks/{task_id}")
            response.raise_for_status()
            return A2AResponse(**response.json())
