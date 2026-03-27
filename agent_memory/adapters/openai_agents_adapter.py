"""
OpenAI Agents SDK Memory Adapter.

Provides lossless memory for OpenAI Agents SDK using Tool and RunHooks.

Usage:
    from agent_memory.adapters import OpenAIAgentsMemory, MemoryTool
    
    # Option 1: Use as a Tool
    memory_tool = MemoryTool(base_path="./my_memory")
    agent = Agent(tools=[memory_tool])
    
    # Option 2: Use as RunHooks for automatic memory
    hooks = OpenAIAgentsMemory(base_path="./my_memory")
    result = Runner.run(agent, "Hello", hooks=hooks)
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from agents import (
        Agent,
        Tool,
        FunctionTool,
        RunContextWrapper,
        RunHooks,
        AgentHooks,
    )
    HAS_OPENAI_AGENTS = True
except ImportError:
    HAS_OPENAI_AGENTS = False
    Tool = object
    RunHooks = object
    AgentHooks = object

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class MemoryTool:
    """
    Memory tool for OpenAI Agents SDK.
    
    Provides remember, recall, and context tools that agents can use
    to interact with the lossless memory system.
    
    Usage:
        from agent_memory.adapters import MemoryTool
        
        memory_tool = MemoryTool(base_path="./my_memory")
        agent = Agent(
            name="Assistant",
            tools=memory_tool.get_tools()
        )
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./openai_agents_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize the memory tool."""
        if not HAS_OPENAI_AGENTS:
            raise ImportError(
                "OpenAI Agents SDK is required. "
                "Install with: pip install openai-agents"
            )
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        logger.info("MemoryTool initialized")
    
    def get_tools(self) -> List[Tool]:
        """Get the list of memory tools for an agent."""
        return [
            self._create_remember_tool(),
            self._create_recall_tool(),
            self._create_get_context_tool(),
        ]
    
    def _create_remember_tool(self) -> Tool:
        """Create the remember tool."""
        async def remember(
            ctx: RunContextWrapper,
            content: str,
            node_type: str = "fact",
            tags: str = "",
            importance: float = 0.5
        ) -> str:
            """Store important information in memory for later retrieval.
            
            Args:
                content: The information to remember
                node_type: Type of memory (fact, experience, decision, error, solution)
                tags: Comma-separated tags for categorization
                importance: Importance score 0-1 (higher = more important)
            """
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            
            node_id = self._adapter.remember(
                content=content,
                node_type=node_type,
                tags=tag_list,
                importance=importance
            )
            
            return f"Stored memory with ID: {node_id}"
        
        return FunctionTool(
            name="remember",
            description="Store important information, decisions, facts, or experiences in long-term memory",
            params_json_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to remember"},
                    "node_type": {"type": "string", "enum": ["fact", "experience", "decision", "error", "solution", "observation"], "default": "fact"},
                    "tags": {"type": "string", "description": "Comma-separated tags"},
                    "importance": {"type": "number", "default": 0.5, "minimum": 0, "maximum": 1}
                },
                "required": ["content"]
            },
            on_invoke_tool=remember,
            strict_json_schema=False
        )
    
    def _create_recall_tool(self) -> Tool:
        """Create the recall tool."""
        async def recall(
            ctx: RunContextWrapper,
            query: str,
            limit: int = 5
        ) -> str:
            """Search memory for relevant information.
            
            Args:
                query: What to search for
                limit: Maximum number of results
            """
            results = self._adapter.recall(query=query, limit=limit)
            
            if not results:
                return "No relevant memories found."
            
            output = f"Found {len(results)} relevant memories:\n\n"
            for i, r in enumerate(results, 1):
                output += f"{i}. [{r.get('node_id', 'N/A')}] {r.get('summary', 'N/A')}\n"
                if r.get('content'):
                    output += f"   Content: {r['content'][:200]}...\n"
                output += f"   Relevance: {r.get('relevance_score', 0):.2f}\n\n"
            
            return output
        
        return FunctionTool(
            name="recall",
            description="Search long-term memory for relevant information",
            params_json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}
                },
                "required": ["query"]
            },
            on_invoke_tool=recall,
            strict_json_schema=False
        )
    
    def _create_get_context_tool(self) -> Tool:
        """Create the get context tool."""
        async def get_memory_context(
            ctx: RunContextWrapper,
            query: str,
            max_tokens: int = 2000
        ) -> str:
            """Get relevant memory context for a topic.
            
            Args:
                query: The topic or question to get context for
                max_tokens: Maximum tokens to include
            """
            result = self._adapter.get_context(query=query, max_tokens=max_tokens)
            
            if result.get("node_count", 0) == 0:
                return "No relevant context found in memory."
            
            return result.get("context", "")
        
        return FunctionTool(
            name="get_memory_context",
            description="Get relevant context from memory for answering a question",
            params_json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to get context for"},
                    "max_tokens": {"type": "integer", "default": 2000, "minimum": 100, "maximum": 8000}
                },
                "required": ["query"]
            },
            on_invoke_tool=get_memory_context,
            strict_json_schema=False
        )
    
    # Direct access methods
    def remember(self, content: str, **kwargs) -> str:
        """Store knowledge directly."""
        return self._adapter.remember(content, **kwargs)
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory directly."""
        return self._adapter.recall(query, limit)
    
    def save(self) -> None:
        """Save memory to disk."""
        self._adapter.save()
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self._adapter.get_stats()


class OpenAIAgentsMemory(RunHooks, AgentHooks):
    """
    OpenAI Agents SDK RunHooks implementation for automatic memory.
    
    Automatically stores and retrieves memories during agent runs.
    
    Usage:
        from agent_memory.adapters import OpenAIAgentsMemory
        
        hooks = OpenAIAgentsMemory(base_path="./my_memory")
        
        # Use with Runner
        result = await Runner.run(
            agent=agent,
            input="Hello",
            hooks=hooks
        )
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./openai_agents_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize the hooks."""
        super().__init__()
        
        if not HAS_OPENAI_AGENTS:
            raise ImportError(
                "OpenAI Agents SDK is required. "
                "Install with: pip install openai-agents"
            )
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        self._current_agent_name: Optional[str] = None
        
        logger.info("OpenAIAgentsMemory hooks initialized")
    
    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        """Called when an agent starts."""
        self._current_agent_name = agent.name if hasattr(agent, 'name') else "unknown"
        logger.debug(f"Agent {self._current_agent_name} starting")
    
    async def on_agent_end(
        self,
        context: RunContextWrapper,
        agent: Agent,
        output: Any
    ) -> None:
        """Called when an agent ends."""
        logger.debug(f"Agent {self._current_agent_name} ended")
        self._adapter.save()
    
    async def on_tool_start(
        self,
        context: RunContextWrapper,
        agent: Agent,
        tool: Tool
    ) -> None:
        """Called when a tool starts."""
        logger.debug(f"Tool {tool.name if hasattr(tool, 'name') else 'unknown'} starting")
    
    async def on_tool_end(
        self,
        context: RunContextWrapper,
        agent: Agent,
        tool: Tool,
        result: str
    ) -> None:
        """Called when a tool ends."""
        tool_name = tool.name if hasattr(tool, 'name') else "unknown"
        
        # Store tool usage
        self._adapter.on_tool_call(
            tool_name=tool_name,
            tool_input={},
            tool_output=result
        )
    
    async def on_handoff(
        self,
        context: RunContextWrapper,
        from_agent: Agent,
        to_agent: Agent
    ) -> None:
        """Called on agent handoff."""
        from_name = from_agent.name if hasattr(from_agent, 'name') else "unknown"
        to_name = to_agent.name if hasattr(to_agent, 'name') else "unknown"
        
        self._adapter.remember(
            content=f"Agent handoff from {from_name} to {to_name}",
            node_type="experience",
            tags=["handoff", from_name, to_name],
            importance=0.3
        )
    
    async def on_llm_start(
        self,
        context: RunContextWrapper,
        agent: Agent,
        system_prompt: str,
        input_items: list
    ) -> None:
        """Called before LLM generation."""
        # Get the user's message for context retrieval
        user_message = ""
        for item in input_items:
            if isinstance(item, dict) and item.get("role") == "user":
                user_message = item.get("content", "")
                break
            elif isinstance(item, str):
                user_message = item
                break
        
        if user_message:
            # Get relevant context
            context_str = self._adapter.before_turn(user_message)
            if context_str:
                logger.debug(f"Injected context for: {user_message[:50]}...")
    
    async def on_llm_end(
        self,
        context: RunContextWrapper,
        agent: Agent,
        response: Any
    ) -> None:
        """Called after LLM generation."""
        # Store the interaction
        if hasattr(response, 'content'):
            self._adapter.after_turn(
                message="",  # We don't have the original message here
                response=str(response.content),
                metadata={"agent": self._current_agent_name}
            )
    
    # Direct access methods
    def remember(self, content: str, **kwargs) -> str:
        """Store knowledge directly."""
        return self._adapter.remember(content, **kwargs)
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory directly."""
        return self._adapter.recall(query, limit)
    
    def get_context(self, query: str, max_tokens: int = 2000) -> Dict:
        """Get formatted context."""
        return self._adapter.get_context(query=query, max_tokens=max_tokens)
    
    def save(self) -> None:
        """Save memory to disk."""
        self._adapter.save()
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self._adapter.get_stats()