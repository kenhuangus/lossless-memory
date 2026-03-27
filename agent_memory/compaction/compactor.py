"""
Compactor - Handles lossless compaction of knowledge nodes.

This module manages the movement of nodes between storage levels,
ensuring no information is lost and indexes are properly updated.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from ..knowledge_graph import KnowledgeNode, StorageLevel, KnowledgeGraph
from ..storage.ram_store import RAMStore
from ..storage.file_store import FileStore
from ..storage.index_manager import IndexManager
from .importance import ImportanceScorer

logger = logging.getLogger(__name__)


class Compactor:
    """
    Manages lossless compaction of knowledge nodes across storage levels.
    
    Compaction Process:
    1. Identify nodes for demotion based on importance scores
    2. Write FULL node content to appropriate file storage
    3. Create/update index entry in parent level
    4. Remove from active RAM but KEEP the index pointer
    5. Update graph references to maintain connectivity
    
    Key Principle: NEVER delete information, only reorganize.
    """
    
    def __init__(
        self,
        ram_store: RAMStore,
        l2_store: FileStore,
        l3_store: FileStore,
        index_manager: IndexManager,
        importance_scorer: ImportanceScorer,
        graph: KnowledgeGraph,
        l1_threshold: float = 0.4,
        l2_threshold: float = 0.2
    ):
        """
        Initialize compactor.
        
        Args:
            ram_store: L1 RAM storage
            l2_store: L2 warm file storage
            l3_store: L3 cold file storage
            index_manager: Global index manager
            importance_scorer: Scorer for node importance
            graph: The knowledge graph
            l1_threshold: Minimum importance to stay in L1
            l2_threshold: Minimum importance to stay in L2 (vs L3)
        """
        self.ram_store = ram_store
        self.l2_store = l2_store
        self.l3_store = l3_store
        self.index_manager = index_manager
        self.scorer = importance_scorer
        self.graph = graph
        
        self.l1_threshold = l1_threshold
        self.l2_threshold = l2_threshold
        
        # Statistics
        self.stats = {
            "l1_to_l2": 0,
            "l2_to_l3": 0,
            "l2_to_l1": 0,  # promotions
            "l3_to_l2": 0,  # promotions
            "total_compacted": 0
        }

    def compact_l1_to_l2(self, max_nodes: int = 10) -> List[str]:
        """
        Compact L1 nodes to L2 storage.
        
        Moves least important L1 nodes to file storage,
        maintaining full content and index entries.
        
        Returns:
            List of node IDs that were compacted
        """
        # Get L1 nodes
        l1_nodes = self.ram_store.get_all_l1_nodes()
        
        if not l1_nodes:
            return []
        
        # Calculate importance scores
        scores = self.scorer.calculate_batch_importance(l1_nodes)
        
        # Find candidates below threshold
        candidates = [
            (node, scores.get(node.id, 0.0))
            for node in l1_nodes
            if scores.get(node.id, 0.0) < self.l1_threshold
        ]
        
        # Sort by importance (lowest first)
        candidates.sort(key=lambda x: x[1])
        
        # Limit to max_nodes
        candidates = candidates[:max_nodes]
        
        compacted_ids = []
        
        for node, score in candidates:
            try:
                # Store to L2 file storage and capture the returned values
                shard_id, byte_offset = self.l2_store.store_node(node)
                
                # Update node metadata
                node.storage_level = StorageLevel.L2_WARM.value
                
                # Update global index using the captured values from store_node
                # Note: node.file_path and node.file_offset are already set by store_node()
                self.index_manager.update_location(
                    node.id,
                    node.file_path,  # This is set by store_node()
                    node.file_offset,  # This is set by store_node()
                    StorageLevel.L2_WARM.value
                )
                
                # Remove from L1 (but keep in index)
                self.ram_store.remove(node.id)
                
                compacted_ids.append(node.id)
                self.stats["l1_to_l2"] += 1
                self.stats["total_compacted"] += 1
                
                logger.debug(
                    f"Compacted node {node.id} to L2 "
                    f"(importance: {score:.3f})"
                )
                
            except Exception as e:
                logger.error(f"Error compacting node {node.id}: {e}")
        
        if compacted_ids:
            logger.info(f"Compacted {len(compacted_ids)} nodes from L1 to L2")
        
        return compacted_ids

    def compact_l2_to_l3(self, max_nodes: int = 20) -> List[str]:
        """
        Compact L2 nodes to L3 cold storage.
        
        Moves least important L2 nodes to cold archive,
        maintaining full content and index entries.
        
        Returns:
            List of node IDs that were compacted
        """
        # Get L2 candidates
        candidates = self.index_manager.get_l2_candidates_for_archive(max_nodes)
        
        compacted_ids = []
        
        for entry in candidates:
            if entry.importance >= self.l2_threshold:
                continue
            
            try:
                # Load node from L2
                node = self.l2_store.load_node(entry.node_id)
                if not node:
                    logger.warning(f"Could not load node {entry.node_id} from L2")
                    continue
                
                # Store to L3 file storage
                shard_id, byte_offset = self.l3_store.store_node(node)
                
                # Update node metadata
                node.storage_level = StorageLevel.L3_COLD.value
                
                # Update global index
                self.index_manager.update_location(
                    node.id,
                    node.file_path,
                    node.file_offset,
                    StorageLevel.L3_COLD.value
                )
                
                # Remove from L2 store index (file content remains)
                self.l2_store.remove_node(node.id)
                
                compacted_ids.append(node.id)
                self.stats["l2_to_l3"] += 1
                self.stats["total_compacted"] += 1
                
                logger.debug(
                    f"Compacted node {node.id} to L3 "
                    f"(importance: {entry.importance:.3f})"
                )
                
            except Exception as e:
                logger.error(f"Error compacting node {entry.node_id}: {e}")
        
        if compacted_ids:
            logger.info(f"Compacted {len(compacted_ids)} nodes from L2 to L3")
        
        return compacted_ids

    def promote_l2_to_l1(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Promote a node from L2 to L1.
        
        Called when a node is accessed and should be in hot memory.
        
        Returns:
            The promoted node or None if not found
        """
        # Load from L2
        node = self.l2_store.load_node(node_id)
        if not node:
            # Check if already in L1
            node = self.ram_store.get(node_id)
            if node:
                return node
            return None
        
        # Update metadata
        node.storage_level = StorageLevel.L1_RAM.value
        node.access()
        
        # Store in L1
        self.ram_store.put(node)
        
        # Update index
        self.index_manager.update_node(node)
        
        self.stats["l2_to_l1"] += 1
        
        logger.debug(f"Promoted node {node_id} from L2 to L1")
        
        return node

    def promote_l3_to_l2(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Promote a node from L3 to L2.
        
        Called when a cold node is accessed.
        
        Returns:
            The promoted node or None if not found
        """
        # Load from L3
        node = self.l3_store.load_node(node_id)
        if not node:
            return None
        
        # Update metadata
        node.storage_level = StorageLevel.L2_WARM.value
        node.access()
        
        # Store in L2
        shard_id, byte_offset = self.l2_store.store_node(node)
        
        # Update index
        self.index_manager.update_node(node)
        
        # Remove from L3 store index
        self.l3_store.remove_node(node_id)
        
        self.stats["l3_to_l2"] += 1
        
        logger.debug(f"Promoted node {node_id} from L3 to L2")
        
        return node

    def promote_to_l1(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Promote a node from any level to L1.
        
        Handles the full promotion path: L3 -> L2 -> L1
        """
        # Check current level
        entry = self.index_manager.get_entry(node_id)
        if not entry:
            return None
        
        if entry.storage_level == StorageLevel.L1_RAM.value:
            return self.ram_store.get(node_id)
        elif entry.storage_level == StorageLevel.L2_WARM.value:
            return self.promote_l2_to_l1(node_id)
        elif entry.storage_level == StorageLevel.L3_COLD.value:
            # Promote L3 -> L2 first
            node = self.promote_l3_to_l2(node_id)
            if node:
                # Then L2 -> L1
                return self.promote_l2_to_l1(node_id)
            return None
        
        return None

    def run_compaction_cycle(self) -> Dict[str, int]:
        """
        Run a full compaction cycle.
        
        This should be called periodically to maintain memory levels.
        
        Returns:
            Dict with compaction statistics
        """
        logger.info("Starting compaction cycle")
        
        # Get current stats
        l1_stats = self.ram_store.get_stats()
        
        # Only compact L1 if above 80% capacity
        l1_compacted = 0
        if l1_stats["utilization"] > 0.8:
            l1_compacted = len(self.compact_l1_to_l2(max_nodes=5))
        
        # Always try to compact L2 -> L3 for very low importance nodes
        l2_compacted = len(self.compact_l2_to_l3(max_nodes=10))
        
        result = {
            "l1_to_l2": l1_compacted,
            "l2_to_l3": l2_compacted,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Compaction cycle complete: {result}")
        
        return result

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Get a node from any storage level.
        
        Handles automatic promotion if node is in cold storage.
        """
        # Check L1 first
        node = self.ram_store.get(node_id)
        if node:
            return node
        
        # Check index for location
        entry = self.index_manager.get_entry(node_id)
        if not entry:
            return None
        
        # Load from appropriate level
        if entry.storage_level == StorageLevel.L2_WARM.value:
            node = self.l2_store.load_node(node_id)
        elif entry.storage_level == StorageLevel.L3_COLD.value:
            node = self.l3_store.load_node(node_id)
        else:
            return None
        
        if node:
            node.access()
            # Update index
            self.index_manager.update_node(node)
        
        return node

    def get_stats(self) -> Dict:
        """Get compaction statistics."""
        return {
            **self.stats,
            "l1_stats": self.ram_store.get_stats(),
            "l2_stats": self.l2_store.get_stats() if self.l2_store else {"total_nodes": 0, "utilization": 0},
            "l3_stats": self.l3_store.get_stats() if self.l3_store else {"total_nodes": 0, "utilization": 0},
            "index_stats": self.index_manager.get_stats()
        }

    def reset_stats(self) -> None:
        """Reset compaction statistics."""
        self.stats = {
            "l1_to_l2": 0,
            "l2_to_l3": 0,
            "l2_to_l1": 0,
            "l3_to_l2": 0,
            "total_compacted": 0
        }