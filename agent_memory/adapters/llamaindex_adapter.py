"""
LlamaIndex Memory Adapter.

Provides lossless memory for LlamaIndex agents and workflows.

Usage:
    from agent_memory.adapters import LlamaIndexMemory
    
    # Use as LlamaIndex BaseMemory
    memory = LlamaIndexMemory(base_path="./my_memory")
    
    agent = OpenAIAgent(tools=tools, memory=memory)
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from llama_index.core.memory import BaseMemory, Memory
    from llama_index.core.llms import ChatMessage, MessageRole
    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False
    BaseMemory = object
    ChatMessage = object
    MessageRole = object

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class LlamaIndexMemory(BaseMemory if HAS_LLAMAINDEX else object):
    """
    LlamaIndex BaseMemory implementation using lossless memory system.
    
    Drop-in replacement for LlamaIndex's default memory.
    
    Usage:
        from agent_memory.adapters import LlamaIndexMemory
        
        memory = LlamaIndexMemory(base_path="./my_memory")
        
        agent = OpenAIAgent(
            tools=[tool1, tool2],
            memory=memory
        )
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./llamaindex_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        token_limit: Optional[int] = None,
    ):
        """Initialize LlamaIndex memory."""
        super().__init__()
        
        if not HAS_LLAMAINDEX:
            raise ImportError(
                "LlamaIndex is required. Install with: pip install llama-index-core"
            )
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        # Keep chat history for compatibility
        self._chat_history: List[ChatMessage] = chat_history or []
        self._token_limit = token_limit
        
        logger.info("LlamaIndexMemory initialized")
    
    @classmethod
    def from_defaults(
        cls,
        chat_history: Optional[List[ChatMessage]] = None,
        token_limit: Optional[int] = None,
        **kwargs
    ) -> "LlamaIndexMemory":
        """Create memory with default settings."""
        return cls(
            chat_history=chat_history,
            token_limit=token_limit,
            **kwargs
        )
    
    def get(self, input: Optional[str] = None, **kwargs) -> List[ChatMessage]:
        """Get chat messages with memory context."""
        messages = list(self._chat_history)
        
        if input:
            # Get relevant context from memory
            context = self._adapter.before_turn(input)
            
            if context:
                # Insert context as system message
                system_msg = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"Relevant context from memory:\n{context}"
                )
                messages = [system_msg] + messages
        
        # Apply token limit if set
        if self.token_limit and len(messages) > 0:
            messages = self._trim_messages(messages)
        
        return messages
    
    def get_all(self) -> List[ChatMessage]:
        """Get all chat messages."""
        return list(self._chat_history)
    
    def put(self, message: ChatMessage) -> None:
        """Put a message into memory."""
        self._chat_history.append(message)
        
        # Store in our memory system
        content = message.content if hasattr(message, 'content') else str(message)
        role = message.role.value if hasattr(message, 'role') else "unknown"
        
        if role == "user":
            self._adapter.remember(
                content=content,
                node_type="experience",
                tags=["llamaindex", "user_message"],
                importance=0.5
            )
        elif role == "assistant":
            self._adapter.remember(
                content=content,
                node_type="experience",
                tags=["llamaindex", "assistant_message"],
                importance=0.4
            )
    
    def set(self, messages: List[ChatMessage]) -> None:
        """Set the chat history."""
        self._chat_history = messages
        
        # Store all messages
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            self._adapter.remember(
                content=content,
                node_type="experience",
                tags=["llamaindex", "history"],
                importance=0.3
            )
    
    def reset(self) -> None:
        """Reset the memory."""
        self._chat_history = []
        # Don't clear our memory system - it's lossless
    
    def _trim_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Trim messages to fit token limit."""
        # Simple estimation: ~4 chars per token
        char_limit = (self.token_limit or 4000) * 4
        
        total_chars = 0
        trimmed = []
        
        # Keep system messages
        for msg in messages:
            if hasattr(msg, 'role') and msg.role == MessageRole.SYSTEM:
                trimmed.append(msg)
                total_chars += len(msg.content) if hasattr(msg, 'content') else 0
        
        # Add recent messages in reverse
        for msg in reversed(messages):
            if hasattr(msg, 'role') and msg.role != MessageRole.SYSTEM:
                msg_len = len(msg.content) if hasattr(msg, 'content') else 0
                if total_chars + msg_len <= char_limit:
                    trimmed.insert(0 if hasattr(msg, 'role') and msg.role == MessageRole.SYSTEM else len([m for m in trimmed if hasattr(m, 'role') and m.role == MessageRole.SYSTEM]), msg)
                    total_chars += msg_len
                else:
                    break
        
        return trimmed
    
    @property
    def token_limit(self) -> Optional[int]:
        """Get token limit."""
        return self._token_limit
    
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


class LlamaIndexChatMemoryBuffer:
    """
    LlamaIndex ChatMemoryBuffer replacement using lossless memory.
    
    Provides a buffer-like interface while maintaining lossless long-term storage.
    
    Usage:
        from agent_memory.adapters import LlamaIndexChatMemoryBuffer
        
        memory = LlamaIndexChatMemoryBuffer(
            base_path="./my_memory",
            token_limit=4000
        )
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./llamaindex_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
        token_limit: int = 4000,
        chat_history: Optional[List[ChatMessage]] = None,
    ):
        """Initialize chat memory buffer."""
        if not HAS_LLAMAINDEX:
            raise ImportError(
                "LlamaIndex is required. Install with: pip install llama-index-core"
            )
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        self._token_limit = token_limit
        self._chat_history: List[ChatMessage] = chat_history or []
        
        logger.info("LlamaIndexChatMemoryBuffer initialized")
    
    @classmethod
    def from_defaults(
        cls,
        token_limit: int = 4000,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs
    ) -> "LlamaIndexChatMemoryBuffer":
        """Create with defaults."""
        return cls(token_limit=token_limit, chat_history=chat_history, **kwargs)
    
    def get(self, input: Optional[str] = None, **kwargs) -> List[ChatMessage]:
        """Get messages with context."""
        messages = list(self._chat_history)
        
        if input:
            context = self._adapter.before_turn(input)
            if context and HAS_LLAMAINDEX:
                system_msg = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"## Memory Context\n{context}"
                )
                messages = [system_msg] + messages
        
        return messages
    
    def get_all(self) -> List[ChatMessage]:
        """Get all messages."""
        return list(self._chat_history)
    
    def put(self, message: ChatMessage) -> None:
        """Add a message."""
        self._chat_history.append(message)
        
        content = message.content if hasattr(message, 'content') else str(message)
        self._adapter.remember(
            content=content,
            node_type="experience",
            tags=["llamaindex", "buffer"],
            importance=0.4
        )
    
    def put_messages(self, messages: List[ChatMessage]) -> None:
        """Add multiple messages."""
        for msg in messages:
            self.put(msg)
    
    def set(self, messages: List[ChatMessage]) -> None:
        """Set messages."""
        self._chat_history = messages
    
    def reset(self) -> None:
        """Reset buffer."""
        self._chat_history = []
    
    @property
    def token_limit(self) -> int:
        """Get token limit."""
        return self._token_limit
    
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