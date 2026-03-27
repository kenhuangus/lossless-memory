"""
Memory Manager - Main API for the multi-level memory system.

This module provides a high-level interface for agents to interact with
the hierarchical memory system, handling storage, retrieval, compaction,
and context assembly.

Supports flexible N-level memory hierarchy (2-7 levels) via configuration.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Set, Union
from datetime import datetime
from pathlib import Path

from .knowledge_graph import (
    KnowledgeNode, 
    KnowledgeGraph, 
    Edge, 
    NodeType, 
    RelationType,
    StorageLevel
)
from .storage import RAMStore, FileStore, IndexManager
from .compaction import Compactor, ImportanceScorer
from .retrieval import MemorySearcher, ContextAssembler
from .level_config import (
    MemoryConfig,
    LevelSpec,
    StorageType,
    get_preset,
    create_assistant_config
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    High-level API for agent memory management.
    
    Features:
    - Simple add/retrieve/search interface
    - Automatic compaction and maintenance
    - Context assembly for LLM prompts
    - Persistent storage across sessions
    - Lossless compaction guarantee
    
    Usage:
        memory = MemoryManager(base_path="./agent_memory_data")
        
        # Store knowledge
        node_id = memory.remember(
            content="The API requires authentication via JWT tokens",
            node_type="fact",
            tags=["api", "authentication"]
        )
        
        # Retrieve knowledge
        results = memory.recall("authentication")
        
        # Get context for LLM
        context = memory.get_context("how to authenticate with API")
    """
    
    def __init__(
        self,
        base_path: str = "./agent_memory_data",
        memory_preset: Optional[str] = None,
        config: Optional[MemoryConfig] = None,
        # Legacy parameters for backward compatibility
        l1_capacity: int = 100,
        l2_shard_size: int = 50,
        l3_shard_size: int = 100,
        auto_compact: bool = True,
        compaction_threshold: float = 0.8
    ):
        """
        Initialize the memory manager with flexible N-level support.
        
        Args:
            base_path: Base directory for persistent storage
            memory_preset: Preset name (chatbot, assistant, enterprise, regulatory, research)
            config: Custom MemoryConfig object (overrides preset)
            l1_capacity: [Legacy] Maximum nodes in L1 RAM
            l2_shard_size: [Legacy] Maximum nodes per L2 shard file
            l3_shard_size: [Legacy] Maximum nodes per L3 shard file
            auto_compact: [Legacy] Enable automatic compaction
            compaction_threshold: [Legacy] L1 utilization threshold to trigger compaction
            
        Examples:
            # Use default 3-level assistant config
            memory = MemoryManager(base_path="./data")
            
            # Use enterprise preset (5 levels)
            memory = MemoryManager(base_path="./data", memory_preset="enterprise")
            
            # Use custom config
            from agent_memory import create_custom_config
            config = create_custom_config(num_levels=4)
            memory = MemoryManager(base_path="./data", config=config)
        """
        self.base_path = Path(base_path)
        
        # Determine configuration
        if config is not None:
            self.config = config
        elif memory_preset is not None:
            self.config = get_preset(memory_preset)
        else:
            # Use legacy 3-level assistant config for backward compatibility
            self.config = create_assistant_config()
            # Override with legacy parameters if provided
            if l1_capacity != 100 or auto_compact != True or compaction_threshold != 0.8:
                self.config.levels[0].capacity = l1_capacity
                self.config.auto_compact = auto_compact
                self.config.compaction_threshold = compaction_threshold
        
        self.auto_compact = self.config.auto_compact
        self.compaction_threshold = self.config.compaction_threshold
        
        # Create directory structure
        self._init_directories()
        
        # Initialize storage layers based on config
        self._init_storage_layers(l2_shard_size, l3_shard_size)
        
        # Initialize index manager
        self.index_manager = IndexManager(
            persistence_path=str(self.base_path / "index.json")
        )
        
        # Initialize knowledge graph
        self.graph = KnowledgeGraph()
        
        # Initialize importance scorer with config decay factor
        self.scorer = ImportanceScorer(decay_factor=self.config.decay_factor)
        
        # Initialize compactor
        self.compactor = Compactor(
            ram_store=self.ram_store,
            l2_store=self.l2_store,
            l3_store=self.l3_store,
            index_manager=self.index_manager,
            importance_scorer=self.scorer,
            graph=self.graph
        )
        
        # Initialize searcher
        self.searcher = MemorySearcher(
            ram_store=self.ram_store,
            l2_store=self.l2_store,
            l3_store=self.l3_store,
            index_manager=self.index_manager,
            compactor=self.compactor,
            graph=self.graph
        )
        
        # Initialize context assembler
        self.assembler = ContextAssembler(
            searcher=self.searcher,
            index_manager=self.index_manager
        )
        
        # Load existing state
        self._load_state()
        
        # Operation counter for auto-compaction
        self._operation_count = 0
        self._compaction_interval = 50  # Compact every N operations
        
        logger.info(
            f"Memory manager initialized: {self.config.num_levels} levels "
            f"(preset: {self.config.name})"
        )
    
    def _init_storage_layers(self, l2_shard_size: int = 50, l3_shard_size: int = 100) -> None:
        """
        Initialize storage layers based on configuration.
        
        For N-level config, we map:
        - Level 0 (hot) -> RAMStore
        - Level 1..N-2 -> FileStore (indexed)
        - Level N-1 (cold) -> FileStore (cold/archive)
        """
        hot_level = self.config.hot_level
        
        # Initialize RAM store for hot level
        self.ram_store = RAMStore(
            max_nodes=hot_level.capacity or 100,
            persistence_path=str(self.base_path / "l1_state.json")
        )
        
        # For backward compatibility and simplicity, we use 2 file stores:
        # - l2_store for warm/intermediate levels
        # - l3_store for cold/archive level
        
        # Determine if we have intermediate levels
        if self.config.num_levels >= 3:
            # Standard 3+ level setup
            self.l2_store = FileStore(
                base_path=str(self.base_path / "l2_storage"),
                level=2,
                max_shard_size=l2_shard_size
            )
            
            self.l3_store = FileStore(
                base_path=str(self.base_path / "l3_storage"),
                level=3,
                max_shard_size=l3_shard_size
            )
        else:
            # 2-level setup: only hot and cold
            self.l2_store = None
            self.l3_store = FileStore(
                base_path=str(self.base_path / "l3_storage"),
                level=1,  # Map to L1 in 2-level system
                max_shard_size=l3_shard_size
            )

    def _init_directories(self) -> None:
        """Create necessary directory structure."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "l2_storage").mkdir(exist_ok=True)
        (self.base_path / "l3_storage").mkdir(exist_ok=True)

    def _load_state(self) -> None:
        """Load existing state from disk."""
        try:
            # Load L1 state
            self.ram_store.load()
            
            # Build index from file stores - load node locations first
            l2_locations = self.l2_store.get_all_node_locations() if self.l2_store else {}
            l3_locations = self.l3_store.get_all_node_locations() if self.l3_store else {}
            
            # Add L1 nodes to graph and index first
            for node in self.ram_store.get_all_l1_nodes():
                self.graph.add_node(node)
                self.index_manager.add_node(node)
            
            # Load L2 nodes directly from file store (not via compactor which depends on index)
            for node_id, (shard_id, offset) in l2_locations.items():
                try:
                    node = self.l2_store.load_node(node_id)
                    if node and node.id not in self.graph.nodes:
                        self.graph.add_node(node)
                        self.index_manager.add_node(node)
                except Exception as e:
                    logger.debug(f"Could not load L2 node {node_id}: {e}")
            
            # Load L3 nodes directly from file store
            for node_id, (shard_id, offset) in l3_locations.items():
                try:
                    node = self.l3_store.load_node(node_id)
                    if node and node.id not in self.graph.nodes:
                        self.graph.add_node(node)
                        self.index_manager.add_node(node)
                except Exception as e:
                    logger.debug(f"Could not load L3 node {node_id}: {e}")
            
            logger.info(
                f"Loaded state: {self.graph.node_count} nodes, "
                f"{self.graph.edge_count} edges"
            )
            
        except Exception as e:
            logger.warning(f"Could not load existing state: {e}")

    def _maybe_compact(self) -> None:
        """Run compaction if needed."""
        if not self.auto_compact:
            return
        
        self._operation_count += 1
        
        if self._operation_count >= self._compaction_interval:
            self._operation_count = 0
            
            stats = self.ram_store.get_stats()
            if stats["utilization"] > self.compaction_threshold:
                self.compactor.run_compaction_cycle()
                
                # Save state after compaction
                self.save()

    def _handle_evicted_node(self, node: KnowledgeNode) -> None:
        """
        Handle a node that was evicted from L1.
        
        Stores the node to L2 to ensure it's not lost.
        """
        try:
            # Store to L2
            shard_id, byte_offset = self.l2_store.store_node(node)
            
            # Update node metadata
            node.storage_level = StorageLevel.L2_WARM.value
            
            # Update index with new location
            self.index_manager.update_location(
                node.id,
                node.file_path,
                node.file_offset,
                StorageLevel.L2_WARM.value
            )
            
            logger.debug(f"Evicted node {node.id} to L2")
        except Exception as e:
            logger.error(f"Error handling evicted node {node.id}: {e}")

    def remember(
        self,
        content: str,
        node_type: str = NodeType.FACT.value,
        tags: List[str] = None,
        summary: str = None,
        importance: float = 0.5,
        metadata: Dict = None
    ) -> str:
        """
        Store a new piece of knowledge.
        
        Args:
            content: The knowledge content
            node_type: Type of knowledge (fact, experience, decision, etc.)
            tags: Tags for categorization
            summary: Optional summary (auto-generated from content if not provided)
            importance: Initial importance (0-1)
            metadata: Additional metadata
            
        Returns:
            Node ID for later retrieval
        """
        # Create node
        node = KnowledgeNode(
            content=content,
            node_type=node_type,
            tags=tags or [],
            summary=summary or content[:200] + "..." if len(content) > 200 else content,
            importance=importance,
            metadata=metadata or {}
        )
        
        # Store in L1 with evicted callback
        self.ram_store.put(node, evicted_callback=self._handle_evicted_node)
        
        # Add to graph
        self.graph.add_node(node)
        
        # Add to index
        self.index_manager.add_node(node)
        
        # Check for auto-compaction
        self._maybe_compact()
        
        logger.debug(f"Remembered node {node.id}: {node.summary[:50]}...")
        
        return node.id

    def remember_with_relation(
        self,
        content: str,
        related_to: str,
        relation_type: str = RelationType.RELATES_TO.value,
        node_type: str = NodeType.FACT.value,
        tags: List[str] = None,
        **kwargs
    ) -> str:
        """
        Store knowledge with a relationship to an existing node.
        
        Args:
            content: The knowledge content
            related_to: ID of the related node
            relation_type: Type of relationship
            node_type: Type of knowledge
            tags: Tags for categorization
            **kwargs: Additional arguments for remember()
            
        Returns:
            New node ID
        """
        # Create the new node
        node_id = self.remember(content, node_type=node_type, tags=tags, **kwargs)
        
        # Create the edge
        if related_to in self.graph.nodes:
            edge = Edge(
                source_id=related_to,
                target_id=node_id,
                relation_type=relation_type
            )
            self.graph.add_edge(edge)
        
        return node_id

    def recall(
        self,
        query: str,
        limit: int = 10,
        load_full: bool = False
    ) -> List[Dict]:
        """
        Recall knowledge matching a query.
        
        Args:
            query: Search query
            limit: Maximum results to return
            load_full: If True, load full node content
            
        Returns:
            List of result dictionaries
        """
        results = self.searcher.search(
            query, 
            limit=limit, 
            load_full=load_full,
            auto_promote=True
        )
        
        return [
            {
                "node_id": r.node_id,
                "summary": r.summary,
                "relevance_score": r.relevance_score,
                "storage_level": r.storage_level,
                "content": r.node.content if r.node else None,
                "was_promoted": r.was_promoted
            }
            for r in results
        ]

    def get(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Get a specific node by ID.
        
        Automatically handles loading from any storage level.
        """
        return self.compactor.get_node(node_id)

    def get_context(
        self,
        query: str,
        max_tokens: int = 4000,
        include_summaries: bool = True,
        include_full_content: bool = True
    ) -> Dict:
        """
        Get formatted context for an LLM prompt.
        
        Args:
            query: The query or topic
            max_tokens: Maximum tokens to include
            include_summaries: Include summaries
            include_full_content: Include full content
            
        Returns:
            Dict with context and metadata
        """
        return self.assembler.assemble_context(
            query,
            max_tokens=max_tokens,
            include_summaries=include_summaries,
            include_full_content=include_full_content
        )

    def get_working_memory(self) -> Dict:
        """Get current working memory (L1) context."""
        return self.assembler.get_working_memory_context()

    def search_by_tag(self, tag: str, limit: int = 10) -> List[Dict]:
        """Search for nodes with a specific tag."""
        results = self.searcher.search_by_tag(tag, limit=limit, load_full=False)
        
        return [
            {
                "node_id": r.node_id,
                "summary": r.summary,
                "relevance_score": r.relevance_score
            }
            for r in results
        ]

    def search_by_type(self, node_type: str, limit: int = 10) -> List[Dict]:
        """Search for nodes of a specific type."""
        results = self.searcher.search_by_type(node_type, limit=limit, load_full=False)
        
        return [
            {
                "node_id": r.node_id,
                "summary": r.summary,
                "relevance_score": r.relevance_score
            }
            for r in results
        ]

    def get_related(self, node_id: str, limit: int = 5) -> List[Dict]:
        """Get nodes related to a given node."""
        results = self.searcher.get_related(node_id, limit=limit, load_full=False)
        
        return [
            {
                "node_id": r.node_id,
                "summary": r.summary,
                "relevance_score": r.relevance_score
            }
            for r in results
        ]

    def get_recent(self, limit: int = 10) -> List[Dict]:
        """Get recently accessed nodes."""
        results = self.searcher.get_recent(limit=limit, load_full=False)
        
        return [
            {
                "node_id": r.node_id,
                "summary": r.summary,
                "relevance_score": r.relevance_score
            }
            for r in results
        ]

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = RelationType.RELATES_TO.value,
        weight: float = 1.0
    ) -> Optional[str]:
        """
        Add a relationship between two existing nodes.
        
        Returns:
            Edge ID or None if nodes don't exist
        """
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            return None
        
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight
        )
        
        return self.graph.add_edge(edge)

    def update_importance(self, node_id: str, importance: float) -> bool:
        """
        Update the importance of a node.
        
        Returns:
            True if successful
        """
        node = self.get(node_id)
        if not node:
            return False
        
        node.importance = max(0.0, min(1.0, importance))
        self.index_manager.update_node(node)
        
        return True

    def force_compact(self) -> Dict:
        """Force a compaction cycle."""
        result = self.compactor.run_compaction_cycle()
        self.save()
        return result

    def save(self) -> None:
        """Save all state to disk."""
        try:
            self.ram_store.save()
            self.index_manager.save()
            logger.info("Memory state saved")
        except Exception as e:
            logger.error(f"Error saving memory state: {e}")

    def get_stats(self) -> Dict:
        """Get comprehensive memory statistics."""
        return {
            "config": {
                "name": self.config.name,
                "num_levels": self.config.num_levels,
                "levels": [
                    {
                        "name": level.name,
                        "level_num": level.level_num,
                        "capacity": level.capacity,
                        "storage_type": level.storage_type.value,
                        "importance_threshold": level.importance_threshold,
                        "retention_period": level.retention_period,
                        "description": level.description
                    }
                    for level in self.config.levels
                ]
            },
            "graph": {
                "nodes": self.graph.node_count,
                "edges": self.graph.edge_count
            },
            "compaction": self.compactor.get_stats(),
            "index": self.index_manager.get_stats()
        }
    
    def get_config(self) -> MemoryConfig:
        """Get current memory configuration."""
        return self.config
    
    @staticmethod
    def list_available_presets() -> Dict[str, str]:
        """
        List all available memory presets.
        
        Returns:
            Dict mapping preset names to descriptions
        """
        from .level_config import list_presets
        return list_presets()

    def get_memory_summary(self) -> str:
        """Get human-readable memory summary."""
        return self.assembler.get_memory_summary()

    def clear(self) -> None:
        """Clear all memory (use with caution!)."""
        self.ram_store.clear()
        self.index_manager.clear()
        
        # Clear file stores
        import shutil
        if (self.base_path / "l2_storage").exists():
            shutil.rmtree(self.base_path / "l2_storage")
        if (self.base_path / "l3_storage").exists():
            shutil.rmtree(self.base_path / "l3_storage")
        
        # Recreate directories
        (self.base_path / "l2_storage").mkdir(exist_ok=True)
        (self.base_path / "l3_storage").mkdir(exist_ok=True)
        
        # Clear graph
        self.graph = KnowledgeGraph()
        
        # Reset compactor graph reference
        self.compactor.graph = self.graph
        self.searcher.graph = self.graph
        
        logger.info("Memory cleared")

    def export_markdown(self, output_path: str) -> None:
        """Export all memory to a markdown file."""
        nodes = []
        
        # Collect all nodes
        for node_id in self.graph.nodes:
            node = self.get(node_id)
            if node:
                nodes.append(node)
        
        # Sort by importance
        nodes.sort(key=lambda n: n.calculate_importance(), reverse=True)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Agent Memory Export\n\n")
            f.write(f"Exported: {datetime.utcnow().isoformat()}\n")
            f.write(f"Total nodes: {len(nodes)}\n\n")
            f.write("---\n\n")
            
            for node in nodes:
                f.write(node.to_markdown())
                f.write("\n\n")
        
        logger.info(f"Memory exported to {output_path}")