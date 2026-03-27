"""
RAM Store (L1) - Hot memory with immediate access.

This is the fastest storage level, keeping active knowledge nodes
in memory with LRU eviction when capacity is exceeded.
"""

import json
import os
from typing import Dict, List, Optional, Set
from datetime import datetime
from collections import OrderedDict

from ..knowledge_graph import KnowledgeNode, StorageLevel


class RAMStore:
    """
    Level 1 storage using RAM with LRU eviction.
    
    Features:
    - O(1) access time
    - LRU eviction when capacity exceeded
    - Keeps summaries of ALL nodes (L1, L2, L3) for fast search
    - Automatic promotion of accessed nodes
    """
    
    def __init__(self, max_nodes: int = 100, persistence_path: Optional[str] = None):
        """
        Initialize RAM store.
        
        Args:
            max_nodes: Maximum number of full nodes to keep in RAM
            persistence_path: Path to persist L1 state (optional)
        """
        self.max_nodes = max_nodes
        self.persistence_path = persistence_path
        
        # Full nodes in L1 (LRU ordered)
        self.nodes: OrderedDict[str, KnowledgeNode] = OrderedDict()
        
        # Index of ALL nodes across all levels (summary only)
        # This allows searching without loading from files
        self.global_index: Dict[str, Dict] = {}
        
        # Track which nodes are in L1 vs indexed elsewhere
        self.l1_node_ids: Set[str] = set()

    def put(self, node: KnowledgeNode, evicted_callback=None) -> None:
        """
        Store a node in L1.
        
        If capacity exceeded, evicts least recently used node.
        
        Args:
            node: The node to store
            evicted_callback: Optional callback function to handle evicted nodes
                            Signature: callback(evicted_node: KnowledgeNode) -> None
        """
        node.storage_level = StorageLevel.L1_RAM.value
        node.access()
        
        # If already in L1, move to end (most recent)
        if node.id in self.nodes:
            self.nodes.move_to_end(node.id)
        else:
            # Evict if at capacity
            while len(self.nodes) >= self.max_nodes:
                evicted_id, evicted_node = self.nodes.popitem(last=False)
                self.l1_node_ids.discard(evicted_id)
                # Call the callback to handle the evicted node
                if evicted_callback:
                    evicted_callback(evicted_node)
        
        self.nodes[node.id] = node
        self.l1_node_ids.add(node.id)
        
        # Update global index
        self._update_index(node)

    def get(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Retrieve a node from L1.
        
        Returns None if node not in L1 (check global_index for location).
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            # Move to end (most recently used)
            self.nodes.move_to_end(node_id)
            node.access()
            self._update_index(node)
            return node
        return None

    def has(self, node_id: str) -> bool:
        """Check if node is in L1."""
        return node_id in self.nodes

    def remove(self, node_id: str) -> Optional[KnowledgeNode]:
        """Remove a node from L1 (for compaction to L2/L3)."""
        if node_id in self.nodes:
            node = self.nodes.pop(node_id)
            self.l1_node_ids.discard(node_id)
            # Keep in global index - just update location
            return node
        return None

    def get_all_l1_nodes(self) -> List[KnowledgeNode]:
        """Get all nodes currently in L1."""
        return list(self.nodes.values())

    def get_l1_node_ids(self) -> Set[str]:
        """Get IDs of all nodes in L1."""
        return self.l1_node_ids.copy()

    def update_index_entry(self, node: KnowledgeNode) -> None:
        """Update the global index for a node (call when node moved to different level)."""
        self._update_index(node)

    def remove_from_index(self, node_id: str) -> None:
        """Remove a node from the global index."""
        self.global_index.pop(node_id, None)

    def search_index(self, query: str) -> List[Dict]:
        """
        Search the global index for nodes matching query.
        
        Returns list of index entries sorted by importance.
        This searches summaries without loading full content.
        """
        query_lower = query.lower()
        matches = []
        
        for node_id, entry in self.global_index.items():
            # Search in summary, tags, and node type
            if (query_lower in entry.get("summary", "").lower() or
                query_lower in entry.get("node_type", "").lower() or
                any(query_lower in tag.lower() for tag in entry.get("tags", []))):
                matches.append(entry)
        
        # Sort by importance
        matches.sort(key=lambda e: e.get("importance", 0), reverse=True)
        return matches

    def get_index_entry(self, node_id: str) -> Optional[Dict]:
        """Get index entry for a node."""
        return self.global_index.get(node_id)

    def get_all_index_entries(self) -> Dict[str, Dict]:
        """Get all index entries."""
        return self.global_index.copy()

    def _update_index(self, node: KnowledgeNode) -> None:
        """Update the global index with node information."""
        self.global_index[node.id] = {
            "id": node.id,
            "summary": node.summary,
            "tags": node.tags,
            "importance": node.calculate_importance(),
            "node_type": node.node_type,
            "storage_level": node.storage_level,
            "file_path": node.file_path,
            "file_offset": node.file_offset,
            "last_accessed": node.last_accessed,
            "created": node.created
        }

    def save(self) -> None:
        """Persist L1 state to disk."""
        if not self.persistence_path:
            return
        
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        
        data = {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "global_index": self.global_index,
            "l1_node_ids": list(self.l1_node_ids)
        }
        
        with open(self.persistence_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        """Load L1 state from disk."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return
        
        with open(self.persistence_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Restore nodes
        self.nodes = OrderedDict()
        for nid, node_data in data.get("nodes", {}).items():
            node = KnowledgeNode.from_dict(node_data)
            self.nodes[nid] = node
        
        # Restore index
        self.global_index = data.get("global_index", {})
        
        # Restore L1 node IDs
        self.l1_node_ids = set(data.get("l1_node_ids", []))

    def clear(self) -> None:
        """Clear all data from RAM store."""
        self.nodes.clear()
        self.global_index.clear()
        self.l1_node_ids.clear()

    @property
    def size(self) -> int:
        """Get number of nodes in L1."""
        return len(self.nodes)

    @property
    def index_size(self) -> int:
        """Get number of entries in global index."""
        return len(self.global_index)

    def get_stats(self) -> Dict:
        """Get statistics about the RAM store."""
        return {
            "l1_nodes": len(self.nodes),
            "max_capacity": self.max_nodes,
            "utilization": len(self.nodes) / self.max_nodes if self.max_nodes > 0 else 0,
            "index_entries": len(self.global_index),
            "l1_node_count": len(self.l1_node_ids)
        }