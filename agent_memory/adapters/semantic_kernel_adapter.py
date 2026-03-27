"""
Semantic Kernel Memory Adapter.

Provides lossless memory for Microsoft Semantic Kernel.

Usage:
    from agent_memory.adapters import SemanticKernelMemory
    
    # Use as Semantic Kernel memory store
    memory = SemanticKernelMemory(base_path="./my_memory")
    
    kernel = Kernel()
    kernel.add_memory_store(memory)
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from semantic_kernel.memory import MemoryStoreBase, MemoryRecord
    from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
    HAS_SEMANTIC_KERNEL = True
except ImportError:
    HAS_SEMANTIC_KERNEL = False
    MemoryStoreBase = object
    MemoryRecord = object

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class SemanticKernelMemoryStore(MemoryStoreBase if HAS_SEMANTIC_KERNEL else object):
    """
    Semantic Kernel MemoryStoreBase implementation.
    
    Drop-in replacement for Semantic Kernel's default memory stores.
    
    Usage:
        from agent_memory.adapters import SemanticKernelMemoryStore
        
        store = SemanticKernelMemoryStore(base_path="./my_memory")
        kernel = Kernel()
        kernel.memory = SemanticTextMemory(store, embeddings)
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./semantic_kernel_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize Semantic Kernel memory store."""
        super().__init__()
        
        if not HAS_SEMANTIC_KERNEL:
            raise ImportError(
                "Semantic Kernel is required. Install with: pip install semantic-kernel"
            )
        
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        logger.info("SemanticKernelMemoryStore initialized")
    
    async def create_collection(self, collection_name: str) -> None:
        """Create a new collection."""
        # Our memory system doesn't need explicit collection creation
        logger.debug(f"Collection '{collection_name}' noted (no-op)")
    
    async def get_collections(self) -> List[str]:
        """Get all collection names."""
        # Return a default collection name
        return ["default"]
    
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        # For lossless memory, we don't actually delete
        logger.debug(f"Collection '{collection_name}' delete requested (no-op)")
    
    async def does_collection_exist(self, collection_name: str) -> bool:
        """Check if collection exists."""
        return True  # Always return True for simplicity
    
    async def upsert(self, collection_name: str, record: MemoryRecord) -> str:
        """Upsert a memory record."""
        content = record.text if hasattr(record, 'text') else str(record)
        
        node_id = self._adapter.remember(
            content=content,
            node_type="fact",
            tags=["semantic_kernel", collection_name],
            importance=0.5,
            metadata={"collection": collection_name, "key": record.key if hasattr(record, 'key') else None}
        )
        
        return node_id
    
    async def upsert_batch(
        self,
        collection_name: str,
        records: List[MemoryRecord]
    ) -> List[str]:
        """Upsert multiple memory records."""
        ids = []
        for record in records:
            node_id = await self.upsert(collection_name, record)
            ids.append(node_id)
        return ids
    
    async def get(
        self,
        collection_name: str,
        key: str,
        with_embedding: bool = False
    ) -> Optional[MemoryRecord]:
        """Get a memory record by key."""
        # Search for the record by key
        results = self._adapter.recall(query=key, limit=1)
        
        if results and HAS_SEMANTIC_KERNEL:
            r = results[0]
            return MemoryRecord(
                id=r.get("node_id", ""),
                text=r.get("content") or r.get("summary", ""),
                key=key,
                timestamp=None,
                embedding=None
            )
        
        return None
    
    async def remove(self, collection_name: str, key: str) -> None:
        """Remove a memory record.
        
        Note: Lossless memory doesn't delete, but marks as archived.
        """
        self._adapter.remember(
            content=f"[ARCHIVED] Record with key: {key}",
            node_type="observation",
            tags=["semantic_kernel", "archived", collection_name],
            importance=0.1
        )
    
    async def get_nearest_matches(
        self,
        collection_name: str,
        embedding: List[float],
        limit: int,
        min_relevance_score: float = 0.0,
        with_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """Get nearest matches by embedding.
        
        Note: Since we use text search, this returns recent/important items.
        """
        results = self._adapter.recall(query="*", limit=limit)
        
        matches = []
        for r in results:
            if r.get("relevance_score", 0) >= min_relevance_score:
                matches.append({
                    "item": r,
                    "score": r.get("relevance_score", 0)
                })
        
        return matches
    
    async def get_nearest_match(
        self,
        collection_name: str,
        embedding: List[float],
        min_relevance_score: float = 0.0,
        with_embedding: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get the nearest match by embedding."""
        matches = await self.get_nearest_matches(
            collection_name,
            embedding,
            limit=1,
            min_relevance_score=min_relevance_score,
            with_embeddings=with_embedding
        )
        
        return matches[0] if matches else None
    
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


class SemanticKernelMemory:
    """
    Semantic Kernel memory adapter with plugin support.
    
    Provides memory functions that can be used as Semantic Kernel plugins.
    
    Usage:
        from agent_memory.adapters import SemanticKernelMemory
        
        memory = SemanticKernelMemory(base_path="./my_memory")
        kernel = Kernel()
        kernel.add_plugin(memory.get_plugin(), plugin_name="memory")
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./semantic_kernel_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize Semantic Kernel memory."""
        self._store = SemanticKernelMemoryStore(
            memory=memory,
            base_path=base_path,
            memory_preset=memory_preset,
            config=config
        )
        
        logger.info("SemanticKernelMemory initialized")
    
    def get_store(self) -> SemanticKernelMemoryStore:
        """Get the memory store."""
        return self._store
    
    def remember(self, content: str, **kwargs) -> str:
        """Store knowledge directly."""
        return self._store.remember(content, **kwargs)
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory directly."""
        return self._store.recall(query, limit)
    
    def save(self) -> None:
        """Save memory to disk."""
        self._store.save()
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self._store.get_stats()