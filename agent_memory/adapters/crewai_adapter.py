"""
CrewAI Memory Adapter.

Provides lossless memory for CrewAI agents and crews.

Usage:
    from agent_memory.adapters import CrewAIMemory
    
    # Use as CrewAI storage
    memory = CrewAIMemory(base_path="./my_memory")
    
    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task1, task2],
        memory=True,
        memory_storage=memory
    )
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from crewai.memory.storage.base_storage import BaseStorage
    from crewai.memory.short_term.short_term_memory import ShortTermMemory
    from crewai.memory.long_term.long_term_memory import LongTermMemory
    from crewai.memory.entity.entity_memory import EntityMemory
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    BaseStorage = object

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class CrewAIStorage(BaseStorage if HAS_CREWAI else object):
    """
    CrewAI BaseStorage implementation using lossless memory.
    
    Drop-in replacement for CrewAI's default memory storage.
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./crewai_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize CrewAI storage."""
        if not HAS_CREWAI:
            raise ImportError(
                "CrewAI is required. Install with: pip install crewai"
            )
        
        super().__init__()
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        logger.info("CrewAIStorage initialized")
    
    def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a memory value."""
        content = str(value) if not isinstance(value, str) else value
        
        self._adapter.remember(
            content=content,
            node_type="experience",
            tags=["crewai", "stored"],
            importance=0.5,
            metadata=metadata
        )
    
    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for memories."""
        results = self._adapter.recall(query=query, limit=limit)
        
        # Filter by score threshold
        return [
            r for r in results
            if r.get("relevance_score", 0) >= score_threshold
        ]
    
    def reset(self) -> None:
        """Reset memory."""
        self._adapter.clear()
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all memories."""
        return self._adapter.recall(query="*", limit=100)


class CrewAIMemory:
    """
    CrewAI memory adapter providing short-term, long-term, and entity memory.
    
    Usage:
        from agent_memory.adapters import CrewAIMemory
        
        memory = CrewAIMemory(base_path="./my_memory")
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            memory=True,
            short_term_memory=memory.short_term,
            long_term_memory=memory.long_term,
            entity_memory=memory.entity
        )
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./crewai_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize CrewAI memory."""
        if not HAS_CREWAI:
            raise ImportError(
                "CrewAI is required. Install with: pip install crewai"
            )
        
        self._base_adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        # Create storage instances
        self._storage = CrewAIStorage(
            memory=self._base_adapter.memory,
            base_path=base_path,
            config=config
        )
        
        # Create memory types
        self.short_term = ShortTermMemory(storage=self._storage)
        self.long_term = LongTermMemory(storage=self._storage)
        self.entity = EntityMemory(storage=self._storage)
        
        logger.info("CrewAIMemory initialized")
    
    def remember(self, content: str, **kwargs) -> str:
        """Store knowledge directly."""
        return self._base_adapter.remember(content, **kwargs)
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory directly."""
        return self._base_adapter.recall(query, limit)
    
    def save(self) -> None:
        """Save memory to disk."""
        self._base_adapter.save()
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self._base_adapter.get_stats()