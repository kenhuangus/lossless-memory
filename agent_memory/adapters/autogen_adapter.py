"""
AutoGen Memory Adapter.

Provides lossless memory for AutoGen agents.

Usage:
    from agent_memory.adapters import AutoGenMemory
    
    # Use with ConversableAgent
    memory = AutoGenMemory(base_path="./my_memory")
    
    agent = ConversableAgent(
        name="Assistant",
        llm_config=llm_config,
        memory=memory
    )
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import ChatMessage, TextMessage
    from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
    HAS_AUTOGEN = True
except ImportError:
    HAS_AUTOGEN = False
    Memory = object
    MemoryContent = object

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class AutoGenMemory(Memory if HAS_AUTOGEN else object):
    """
    AutoGen Memory implementation using lossless memory system.
    
    Implements the AutoGen Memory interface for seamless integration
    with AutoGen agents.
    
    Usage:
        from agent_memory.adapters import AutoGenMemory
        
        memory = AutoGenMemory(base_path="./my_memory")
        
        agent = AssistantAgent(
            name="Assistant",
            llm_config=llm_config,
            memory=[memory]
        )
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./autogen_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize AutoGen memory."""
        super().__init__()
        
        if not HAS_AUTOGEN:
            raise ImportError(
                "AutoGen is required. Install with: pip install autogen-agentchat"
            )
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        logger.info("AutoGenMemory initialized")
    
    async def add(self, content: MemoryContent, **kwargs) -> None:
        """Add a memory content."""
        text = content.content if hasattr(content, 'content') else str(content)
        
        self._adapter.remember(
            content=text,
            node_type="experience",
            tags=["autogen", "added"],
            importance=0.5
        )
    
    async def query(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[MemoryContent]:
        """Query memory for relevant content."""
        results = self._adapter.recall(query=query, limit=top_k)
        
        memory_contents = []
        for r in results:
            content = r.get("content") or r.get("summary", "")
            if content:
                if HAS_AUTOGEN:
                    mc = MemoryContent(
                        content=content,
                        mime_type=MemoryMimeType.TEXT
                    )
                    memory_contents.append(mc)
        
        return memory_contents
    
    async def update(self, content: MemoryContent, **kwargs) -> None:
        """Update a memory content."""
        # For lossless memory, we just add as new
        await self.add(content, **kwargs)
    
    async def delete(self, content: MemoryContent, **kwargs) -> None:
        """Delete a memory content.
        
        Note: Lossless memory doesn't delete, but we mark it as archived.
        """
        text = content.content if hasattr(content, 'content') else str(content)
        
        self._adapter.remember(
            content=f"[ARCHIVED] {text}",
            node_type="observation",
            tags=["autogen", "archived"],
            importance=0.1
        )
    
    async def clear(self) -> None:
        """Clear all memory."""
        self._adapter.clear()
    
    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get context string for a query."""
        result = self._adapter.get_context(query=query, max_tokens=max_tokens)
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


class AutoGenConversableMemory:
    """
    Memory adapter for AutoGen ConversableAgent.
    
    Provides a simpler interface for older AutoGen versions.
    
    Usage:
        from agent_memory.adapters import AutoGenConversableMemory
        
        memory = AutoGenConversableMemory(base_path="./my_memory")
        
        agent = ConversableAgent(
            name="Assistant",
            llm_config=llm_config
        )
        
        # Register memory hooks
        memory.register_with_agent(agent)
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./autogen_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize ConversableAgent memory."""
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        logger.info("AutoGenConversableMemory initialized")
    
    def register_with_agent(self, agent: Any) -> None:
        """Register memory hooks with a ConversableAgent."""
        if not HAS_AUTOGEN:
            raise ImportError(
                "AutoGen is required. Install with: pip install autogen-agentchat"
            )
        
        # Store reference to original methods
        original_generate_reply = agent.generate_reply
        
        async def memory_augmented_generate_reply(
            messages: Optional[List[Dict]] = None,
            sender: Optional[Any] = None,
            **kwargs
        ):
            """Generate reply with memory context."""
            # Get user message
            user_message = ""
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    user_message = last_msg.get("content", "")
                else:
                    user_message = str(last_msg)
            
            # Get memory context
            context = self._adapter.before_turn(user_message)
            
            # Add context to system message if available
            if context:
                # Prepend context to the conversation
                if messages is None:
                    messages = []
                messages = [{"role": "system", "content": context}] + messages
            
            # Call original method
            response = await original_generate_reply(messages, sender, **kwargs)
            
            # Store the interaction
            if response:
                response_text = ""
                if isinstance(response, dict):
                    response_text = response.get("content", str(response))
                else:
                    response_text = str(response)
                
                self._adapter.after_turn(
                    message=user_message,
                    response=response_text
                )
            
            return response
        
        agent.generate_reply = memory_augmented_generate_reply
    
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