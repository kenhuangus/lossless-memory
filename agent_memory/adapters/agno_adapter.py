"""
Agno Memory Adapter.

Provides lossless memory for Agno agents.

Usage:
    from agent_memory.adapters import AgnoMemory
    
    # Use as Agno memory
    memory = AgnoMemory(base_path="./my_memory")
    
    agent = Agent(
        name="Assistant",
        memory=memory
    )
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from agno.memory import Memory, MemoryRow
    from agno.memory.db import MemoryDb
    HAS_AGNO = True
except ImportError:
    HAS_AGNO = False
    Memory = object
    MemoryDb = object
    MemoryRow = object

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class AgnoMemoryDb(MemoryDb if HAS_AGNO else object):
    """
    Agno MemoryDb implementation using lossless memory system.
    
    Drop-in replacement for Agno's default memory databases.
    
    Usage:
        from agent_memory.adapters import AgnoMemoryDb
        
        db = AgnoMemoryDb(base_path="./my_memory")
        memory = Memory(db=db)
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./agno_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize Agno memory database."""
        super().__init__()
        
        if not HAS_AGNO:
            raise ImportError(
                "Agno is required. Install with: pip install agno"
            )
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        logger.info("AgnoMemoryDb initialized")
    
    def create(self) -> None:
        """Create the database."""
        # No-op for our system
        pass
    
    def memory_exists(self, memory_row: MemoryRow) -> bool:
        """Check if a memory exists."""
        # Search for similar content
        content = memory_row.memory if hasattr(memory_row, 'memory') else str(memory_row)
        results = self._adapter.recall(query=content[:100], limit=1)
        
        if results:
            existing = results[0].get("content") or results[0].get("summary", "")
            return existing[:100] == content[:100]
        
        return False
    
    def read_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None
    ) -> List[MemoryRow]:
        """Read memories."""
        query = f"user:{user_id}" if user_id else "*"
        results = self._adapter.recall(query=query, limit=limit or 100)
        
        memories = []
        for r in results:
            if HAS_AGNO:
                row = MemoryRow(
                    memory=r.get("content") or r.get("summary", ""),
                    user_id=user_id or "",
                    created_at=r.get("created_at", "")
                )
                memories.append(row)
        
        return memories
    
    def upsert_memory(self, memory_row: MemoryRow) -> Optional[MemoryRow]:
        """Upsert a memory."""
        content = memory_row.memory if hasattr(memory_row, 'memory') else str(memory_row)
        user_id = memory_row.user_id if hasattr(memory_row, 'user_id') else None
        
        node_id = self._adapter.remember(
            content=content,
            node_type="experience",
            tags=["agno", f"user:{user_id}"] if user_id else ["agno"],
            importance=0.5
        )
        
        return memory_row
    
    def delete_memory(self, memory_row: MemoryRow) -> None:
        """Delete a memory.
        
        Note: Lossless memory doesn't delete, but marks as archived.
        """
        content = memory_row.memory if hasattr(memory_row, 'memory') else str(memory_row)
        
        self._adapter.remember(
            content=f"[ARCHIVED] {content}",
            node_type="observation",
            tags=["agno", "archived"],
            importance=0.1
        )
    
    def drop_table(self) -> None:
        """Drop the table."""
        # For lossless memory, we don't actually drop
        logger.debug("Drop table requested (no-op for lossless memory)")
    
    def table_exists(self) -> bool:
        """Check if table exists."""
        return True
    
    def clear(self) -> None:
        """Clear all memory."""
        self._adapter.clear()
    
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


class AgnoMemory(Memory if HAS_AGNO else object):
    """
    Agno Memory implementation using lossless memory system.
    
    Drop-in replacement for Agno's default memory.
    
    Usage:
        from agent_memory.adapters import AgnoMemory
        
        memory = AgnoMemory(base_path="./my_memory")
        
        agent = Agent(
            name="Assistant",
            memory=memory
        )
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./agno_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
        db: Optional[AgnoMemoryDb] = None,
    ):
        """Initialize Agno memory."""
        if not HAS_AGNO:
            raise ImportError(
                "Agno is required. Install with: pip install agno"
            )
        
        if db is not None:
            self.db = db
        else:
            self.db = AgnoMemoryDb(
                memory=memory,
                base_path=base_path,
                memory_preset=memory_preset,
                config=config
            )
        
        super().__init__(db=self.db)
        
        logger.info("AgnoMemory initialized")
    
    def remember(self, content: str, **kwargs) -> str:
        """Store knowledge directly."""
        return self.db.remember(content, **kwargs)
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory directly."""
        return self.db.recall(query, limit)
    
    def save(self) -> None:
        """Save memory to disk."""
        self.db.save()
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self.db.get_stats()


class AgnoAssistantMemory:
    """
    Simple Agno memory adapter for assistant-style agents.
    
    Usage:
        from agent_memory.adapters import AgnoAssistantMemory
        
        memory = AgnoAssistantMemory(base_path="./my_memory")
        
        # Use with agent
        agent = Agent(
            name="Assistant",
            add_memory=memory
        )
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./agno_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize assistant memory."""
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        logger.info("AgnoAssistantMemory initialized")
    
    def add(self, content: str, user_id: Optional[str] = None, **kwargs) -> str:
        """Add a memory."""
        tags = ["agno", "assistant"]
        if user_id:
            tags.append(f"user:{user_id}")
        
        return self._adapter.remember(
            content=content,
            node_type="experience",
            tags=tags,
            importance=0.5
        )
    
    def search(self, query: str, user_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Search memories."""
        if user_id:
            query = f"{query} user:{user_id}"
        return self._adapter.recall(query=query, limit=limit)
    
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get context for a query."""
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