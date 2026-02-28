"""
A2A (Agent-to-Agent) Protocol Models.
Simplified implementation inspired by Google's A2A protocol.
Defines Agent Cards, Task models, and structured request/response types.
"""
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --- Agent Card ---

class AgentCard(BaseModel):
    """Metadata describing an agent's capabilities and endpoint."""
    name: str
    description: str
    url: str
    capabilities: list[str]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


# --- Task Lifecycle ---

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class A2ATask(BaseModel):
    """A unit of work delegated from the Orchestrator to an Agent."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    capability: str  # Which capability to invoke (e.g., "ingest_video")
    input_data: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class A2AResponse(BaseModel):
    """Structured response from an agent after processing a task."""
    task_id: str
    status: str  # "success" or "failure"
    result: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# --- MCP-Compliant Response (for Agent 2) ---

class MCPSuccessResponse(BaseModel):
    """MCP-compliant success response for product search."""
    status: str = "success"
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str = "Request processed successfully"
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MCPFailureResponse(BaseModel):
    """MCP-compliant failure response."""
    status: str = "failure"
    error_code: str
    message: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# --- Orchestrator Chat ---

class ChatRequest(BaseModel):
    """Request from Streamlit UI to the Orchestrator."""
    message: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chat_history: list[dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response from the Orchestrator back to Streamlit UI."""
    message: str
    agent_used: str | None = None
    products: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
