"""
Haystack Memory Adapter.

Provides lossless memory for Haystack pipelines and agents.

Usage:
    from agent_memory.adapters import HaystackMemory
    
    # Use as Haystack component
    memory = HaystackMemory(base_path="./my_memory")
    
    pipeline = Pipeline()
    pipeline.add_component("memory", memory)
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from haystack import component, Document
    from haystack.components.memory import Memory
    HAS_HAYSTACK = True
except ImportError:
    HAS_HAYSTACK = False
    # Create a mock component decorator that works without haystack
    class _MockComponentDecorator:
        def __call__(self, cls):
            return cls
        def output_types(self, **kwargs):
            def decorator(cls):
                return cls
            return decorator
    component = _MockComponentDecorator()
    Document = object
    Memory = object

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class HaystackMemoryComponent:
    """
    Haystack component for memory operations.
    
    Provides remember, recall, and context as Haystack components.
    
    Usage:
        from agent_memory.adapters import HaystackMemoryComponent
        
        memory = HaystackMemoryComponent(base_path="./my_memory")
        
        pipeline = Pipeline()
        pipeline.add_component("memory", memory)
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./haystack_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize Haystack memory component."""
        if not HAS_HAYSTACK:
            raise ImportError(
                "Haystack is required. Install with: pip install haystack-ai"
            )
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        logger.info("HaystackMemoryComponent initialized")
    
    @component.output_types(documents=List[Document], context=str)
    def run(
        self,
        query: str,
        operation: str = "recall",
        content: Optional[str] = None,
        node_type: str = "fact",
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        limit: int = 5,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Run memory operation.
        
        Args:
            query: Query for recall/context operations
            operation: Operation type (remember, recall, context)
            content: Content to remember (for remember operation)
            node_type: Node type for remember operation
            tags: Tags for remember operation
            importance: Importance for remember operation
            limit: Limit for recall operation
            max_tokens: Max tokens for context operation
            
        Returns:
            Dict with documents (for recall) or context (for context)
        """
        if operation == "remember" and content:
            node_id = self._adapter.remember(
                content=content,
                node_type=node_type,
                tags=tags or [],
                importance=importance
            )
            return {
                "documents": [],
                "context": f"Stored memory with ID: {node_id}"
            }
        
        elif operation == "recall":
            results = self._adapter.recall(query=query, limit=limit)
            
            documents = []
            for r in results:
                content = r.get("content") or r.get("summary", "")
                if content and HAS_HAYSTACK:
                    doc = Document(
                        content=content,
                        meta={
                            "node_id": r.get("node_id", ""),
                            "relevance_score": r.get("relevance_score", 0),
                            "storage_level": r.get("storage_level", "")
                        }
                    )
                    documents.append(doc)
            
            return {"documents": documents, "context": ""}
        
        elif operation == "context":
            result = self._adapter.get_context(
                query=query,
                max_tokens=max_tokens
            )
            return {
                "documents": [],
                "context": result.get("context", "")
            }
        
        else:
            return {"documents": [], "context": ""}
    
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


class HaystackMemory(Memory if HAS_HAYSTACK else object):
    """
    Haystack Memory implementation using lossless memory system.
    
    Drop-in replacement for Haystack's default memory.
    
    Usage:
        from agent_memory.adapters import HaystackMemory
        
        memory = HaystackMemory(base_path="./my_memory")
        
        agent = Agent(
            chat_generator=generator,
            memory=memory
        )
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./haystack_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
        window_size: int = 10,
    ):
        """Initialize Haystack memory."""
        super().__init__()
        
        if not HAS_HAYSTACK:
            raise ImportError(
                "Haystack is required. Install with: pip install haystack-ai"
            )
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        self._window_size = window_size
        self._messages: List[Dict[str, str]] = []
        
        logger.info("HaystackMemory initialized")
    
    def get(
        self,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Get messages as documents."""
        messages = self._messages
        
        if top_k:
            messages = messages[-top_k:]
        
        if not HAS_HAYSTACK:
            return []
        
        documents = []
        for msg in messages:
            doc = Document(
                content=msg.get("content", ""),
                meta={
                    "role": msg.get("role", ""),
                    "timestamp": msg.get("timestamp", "")
                }
            )
            documents.append(doc)
        
        return documents
    
    def add(
        self,
        message: Dict[str, str],
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a message."""
        self._messages.append(message)
        
        # Keep only window size
        if len(self._messages) > self._window_size:
            self._messages = self._messages[-self._window_size:]
        
        # Store in long-term memory
        content = message.get("content", "")
        role = message.get("role", "unknown")
        
        self._adapter.remember(
            content=content,
            node_type="experience",
            tags=["haystack", role],
            importance=0.5,
            metadata=metadata
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """Search memory."""
        results = self._adapter.recall(query=query, limit=top_k)
        
        if not HAS_HAYSTACK:
            return []
        
        documents = []
        for r in results:
            content = r.get("content") or r.get("summary", "")
            if content:
                doc = Document(
                    content=content,
                    meta={
                        "node_id": r.get("node_id", ""),
                        "relevance_score": r.get("relevance_score", 0)
                    }
                )
                documents.append(doc)
        
        return documents
    
    def delete(self, document_ids: List[str]) -> None:
        """Delete documents.
        
        Note: Lossless memory doesn't delete, but marks as archived.
        """
        for doc_id in document_ids:
            self._adapter.remember(
                content=f"[ARCHIVED] Document ID: {doc_id}",
                node_type="observation",
                tags=["haystack", "archived"],
                importance=0.1
            )
    
    def count(self) -> int:
        """Count messages."""
        return len(self._messages)
    
    def filter(
        self,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Filter messages."""
        return self.get(filters=filters, top_k=top_k)
    
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


class HaystackChatMemory:
    """
    Chat-style memory for Haystack agents.
    
    Usage:
        from agent_memory.adapters import HaystackChatMemory
        
        memory = HaystackChatMemory(base_path="./my_memory")
        
        # Use in pipeline
        pipeline = Pipeline()
        pipeline.add_component("memory", memory)
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./haystack_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize chat memory."""
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        self._messages: List[Dict[str, str]] = []
        
        logger.info("HaystackChatMemory initialized")
    
    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self._messages.append({"role": "user", "content": content})
        self._adapter.remember(
            content=content,
            node_type="experience",
            tags=["haystack", "user"],
            importance=0.5
        )
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self._messages.append({"role": "assistant", "content": content})
        self._adapter.remember(
            content=content,
            node_type="experience",
            tags=["haystack", "assistant"],
            importance=0.4
        )
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return list(self._messages)
    
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get context for a query."""
        result = self._adapter.get_context(query=query, max_tokens=max_tokens)
        return result.get("context", "")
    
    def clear(self) -> None:
        """Clear recent messages."""
        self._messages = []
    
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