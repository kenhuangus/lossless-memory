"""
Pydantic AI Memory Adapter.

Provides lossless memory for Pydantic AI agents.

Usage:
    from agent_memory.adapters import PydanticAIMemory
    
    # Use as Pydantic AI memory
    memory = PydanticAIMemory(base_path="./my_memory")
    
    agent = Agent(
        model="openai:gpt-4",
        deps_type=MyDeps,
        memory=memory
    )
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from pydantic_ai import Agent
    from pydantic_ai.memory import Memory, SimpleMemory
    HAS_PYDANTIC_AI = True
except ImportError:
    HAS_PYDANTIC_AI = False
    Memory = object

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class PydanticAIMemory(Memory if HAS_PYDANTIC_AI else object):
    """
    Pydantic AI Memory implementation using lossless memory system.
    
    Drop-in replacement for Pydantic AI's default memory.
    
    Usage:
        from agent_memory.adapters import PydanticAIMemory
        
        memory = PydanticAIMemory(base_path="./my_memory")
        
        agent = Agent(
            model="openai:gpt-4",
            memory=memory
        )
        
        result = agent.run_sync("Hello!", memory=memory)
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./pydantic_ai_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize Pydantic AI memory."""
        super().__init__()
        
        if not HAS_PYDANTIC_AI:
            raise ImportError(
                "Pydantic AI is required. Install with: pip install pydantic-ai"
            )
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        # Keep recent messages for context window
        self._recent_messages: List[Dict[str, str]] = []
        self._max_recent = 20
        
        logger.info("PydanticAIMemory initialized")
    
    def add_message(
        self,
        role: str,
        content: str,
        **kwargs
    ) -> None:
        """Add a message to memory."""
        # Store in recent messages
        self._recent_messages.append({"role": role, "content": content})
        
        # Keep only recent messages
        if len(self._recent_messages) > self._max_recent:
            self._recent_messages = self._recent_messages[-self._max_recent:]
        
        # Store in long-term memory
        if role == "user":
            self._adapter.remember(
                content=content,
                node_type="experience",
                tags=["pydantic_ai", "user_message"],
                importance=0.5
            )
        elif role == "assistant":
            self._adapter.remember(
                content=content,
                node_type="experience",
                tags=["pydantic_ai", "assistant_message"],
                importance=0.4
            )
        elif role == "tool":
            self._adapter.on_tool_call(
                tool_name=kwargs.get("tool_name", "unknown"),
                tool_input=kwargs.get("tool_input", {}),
                tool_output=content
            )
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages from memory."""
        if limit:
            return self._recent_messages[-limit:]
        return list(self._recent_messages)
    
    def get_context_for_prompt(self, query: str = "") -> str:
        """Get context string for inclusion in prompts."""
        if query:
            result = self._adapter.get_context(query=query, max_tokens=2000)
            return result.get("context", "")
        return ""
    
    def clear(self) -> None:
        """Clear recent messages."""
        self._recent_messages = []
        # Don't clear long-term memory - it's lossless
    
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


class PydanticAIToolMemory:
    """
    Memory tool for Pydantic AI agents.
    
    Provides remember, recall, and context as tools.
    
    Usage:
        from agent_memory.adapters import PydanticAIToolMemory
        
        tool_memory = PydanticAIToolMemory(base_path="./my_memory")
        
        @agent.tool
        async def remember(ctx, content: str, importance: float = 0.5) -> str:
            return await tool_memory.remember_tool(content, importance)
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./pydantic_ai_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize tool memory."""
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        logger.info("PydanticAIToolMemory initialized")
    
    async def remember_tool(
        self,
        content: str,
        node_type: str = "fact",
        tags: str = "",
        importance: float = 0.5
    ) -> str:
        """Remember tool function."""
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        
        node_id = self._adapter.remember(
            content=content,
            node_type=node_type,
            tags=tag_list,
            importance=importance
        )
        
        return f"Stored memory with ID: {node_id}"
    
    async def recall_tool(self, query: str, limit: int = 5) -> str:
        """Recall tool function."""
        results = self._adapter.recall(query=query, limit=limit)
        
        if not results:
            return "No relevant memories found."
        
        output = f"Found {len(results)} relevant memories:\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('summary', 'N/A')}\n"
            if r.get('content'):
                output += f"   Content: {r['content'][:200]}...\n"
            output += f"   Relevance: {r.get('relevance_score', 0):.2f}\n\n"
        
        return output
    
    async def context_tool(self, query: str, max_tokens: int = 2000) -> str:
        """Context tool function."""
        result = self._adapter.get_context(query=query, max_tokens=max_tokens)
        
        if result.get("node_count", 0) == 0:
            return "No relevant context found."
        
        return result.get("context", "")
    
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


class PydanticAIDepsMemory:
    """
    Memory as a dependency for Pydantic AI agents.
    
    Use with deps_type for type-safe memory access.
    
    Usage:
        from dataclasses import dataclass
        from agent_memory.adapters import PydanticAIDepsMemory
        
        @dataclass
        class MyDeps:
            memory: PydanticAIDepsMemory
            # other deps...
        
        agent = Agent(
            model="openai:gpt-4",
            deps_type=MyDeps
        )
        
        deps = MyDeps(memory=PydanticAIDepsMemory(base_path="./my_memory"))
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./pydantic_ai_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize deps memory."""
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        logger.info("PydanticAIDepsMemory initialized")
    
    def remember(self, content: str, **kwargs) -> str:
        """Store knowledge."""
        return self._adapter.remember(content, **kwargs)
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory."""
        return self._adapter.recall(query, limit)
    
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get context for a query."""
        result = self._adapter.get_context(query=query, max_tokens=max_tokens)
        return result.get("context", "")
    
    def before_turn(self, message: str) -> Optional[str]:
        """Get context before a turn."""
        return self._adapter.before_turn(message)
    
    def after_turn(self, message: str, response: str, **kwargs) -> None:
        """Store after a turn."""
        self._adapter.after_turn(message, response, **kwargs)
    
    def save(self) -> None:
        """Save memory to disk."""
        self._adapter.save()
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self._adapter.get_stats()