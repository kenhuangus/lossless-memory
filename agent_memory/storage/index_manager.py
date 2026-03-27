"""
Index Manager - Manages cross-level indexes for fast retrieval.

This module provides a unified index across all storage levels,
enabling fast lookups without loading full node content.
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path

from ..knowledge_graph import KnowledgeNode, StorageLevel


class IndexEntry:
    """
    Compact representation of a node for index storage.
    
    Contains enough information to locate and retrieve the full node.
    """
    
    def __init__(self, node: KnowledgeNode = None):
        self.node_id: str = ""
        self.summary: str = ""
        self.tags: List[str] = []
        self.importance: float = 0.0
        self.node_type: str = ""
        self.storage_level: int = StorageLevel.L1_RAM.value
        self.file_path: Optional[str] = None
        self.file_offset: Optional[int] = None
        self.last_accessed: str = ""
        self.created: str = ""
        
        if node:
            self.from_node(node)

    def from_node(self, node: KnowledgeNode) -> None:
        """Populate from a KnowledgeNode."""
        self.node_id = node.id
        self.summary = node.summary
        self.tags = node.tags.copy()
        self.importance = node.calculate_importance()
        self.node_type = node.node_type
        self.storage_level = node.storage_level
        self.file_path = node.file_path
        self.file_offset = node.file_offset
        self.last_accessed = node.last_accessed
        self.created = node.created

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "summary": self.summary,
            "tags": self.tags,
            "importance": self.importance,
            "node_type": self.node_type,
            "storage_level": self.storage_level,
            "file_path": self.file_path,
            "file_offset": self.file_offset,
            "last_accessed": self.last_accessed,
            "created": self.created
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'IndexEntry':
        """Deserialize from dictionary."""
        entry = cls()
        entry.node_id = data.get("node_id", "")
        entry.summary = data.get("summary", "")
        entry.tags = data.get("tags", [])
        entry.importance = data.get("importance", 0.0)
        entry.node_type = data.get("node_type", "")
        entry.storage_level = data.get("storage_level", StorageLevel.L1_RAM.value)
        entry.file_path = data.get("file_path")
        entry.file_offset = data.get("file_offset")
        entry.last_accessed = data.get("last_accessed", "")
        entry.created = data.get("created", "")
        return entry

    def matches_query(self, query: str) -> bool:
        """Check if this entry matches a search query."""
        query_lower = query.lower()
        
        # Check summary
        if query_lower in self.summary.lower():
            return True
        
        # Check tags
        if any(query_lower in tag.lower() for tag in self.tags):
            return True
        
        # Check node type
        if query_lower in self.node_type.lower():
            return True
        
        return False


class IndexManager:
    """
    Manages unified indexes across all storage levels.
    
    Features:
    - Maintains index of ALL nodes regardless of storage level
    - Supports text search across summaries and tags
    - Tracks node locations for retrieval
    - Persists index to disk for recovery
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        """
        Initialize index manager.
        
        Args:
            persistence_path: Path to persist index state
        """
        self.persistence_path = persistence_path
        
        # Main index: node_id -> IndexEntry
        self.entries: Dict[str, IndexEntry] = {}
        
        # Tag index: tag -> set of node_ids
        self.tag_index: Dict[str, Set[str]] = {}
        
        # Type index: node_type -> set of node_ids
        self.type_index: Dict[str, Set[str]] = {}
        
        # Level index: storage_level -> set of node_ids
        self.level_index: Dict[int, Set[str]] = {
            StorageLevel.L1_RAM.value: set(),
            StorageLevel.L2_WARM.value: set(),
            StorageLevel.L3_COLD.value: set()
        }
        
        # Load existing index if available
        if persistence_path:
            self.load()

    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node to the index."""
        entry = IndexEntry(node)
        self.entries[node.id] = entry
        
        # Update tag index
        for tag in node.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(node.id)
        
        # Update type index
        if node.node_type not in self.type_index:
            self.type_index[node.node_type] = set()
        self.type_index[node.node_type].add(node.id)
        
        # Update level index
        if node.storage_level in self.level_index:
            self.level_index[node.storage_level].add(node.id)

    def update_node(self, node: KnowledgeNode) -> None:
        """Update an existing node in the index."""
        # Remove old entries first
        old_entry = self.entries.get(node.id)
        if old_entry:
            # Remove from tag index
            for tag in old_entry.tags:
                if tag in self.tag_index:
                    self.tag_index[tag].discard(node.id)
            
            # Remove from type index
            if old_entry.node_type in self.type_index:
                self.type_index[old_entry.node_type].discard(node.id)
            
            # Remove from level index
            if old_entry.storage_level in self.level_index:
                self.level_index[old_entry.storage_level].discard(node.id)
        
        # Add updated entry
        self.add_node(node)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the index."""
        if node_id not in self.entries:
            return False
        
        entry = self.entries.pop(node_id)
        
        # Remove from tag index
        for tag in entry.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(node_id)
        
        # Remove from type index
        if entry.node_type in self.type_index:
            self.type_index[entry.node_type].discard(node_id)
        
        # Remove from level index
        if entry.storage_level in self.level_index:
            self.level_index[entry.storage_level].discard(node_id)
        
        return True

    def get_entry(self, node_id: str) -> Optional[IndexEntry]:
        """Get index entry for a node."""
        return self.entries.get(node_id)

    def search(self, query: str, limit: int = 20) -> List[IndexEntry]:
        """
        Search the index for nodes matching query.
        
        Returns entries sorted by importance.
        """
        matches = [
            entry for entry in self.entries.values()
            if entry.matches_query(query)
        ]
        
        # Sort by importance (descending)
        matches.sort(key=lambda e: e.importance, reverse=True)
        
        return matches[:limit]

    def search_by_tag(self, tag: str) -> List[IndexEntry]:
        """Find all nodes with a specific tag."""
        node_ids = self.tag_index.get(tag, set())
        return [self.entries[nid] for nid in node_ids if nid in self.entries]

    def search_by_type(self, node_type: str) -> List[IndexEntry]:
        """Find all nodes of a specific type."""
        node_ids = self.type_index.get(node_type, set())
        return [self.entries[nid] for nid in node_ids if nid in self.entries]

    def get_by_level(self, level: int) -> List[IndexEntry]:
        """Get all nodes at a specific storage level."""
        node_ids = self.level_index.get(level, set())
        return [self.entries[nid] for nid in node_ids if nid in self.entries]

    def get_l1_candidates_for_eviction(self, count: int = 10) -> List[IndexEntry]:
        """
        Get L1 nodes that are candidates for eviction to L2.
        
        Selects based on importance and last access time.
        """
        l1_entries = self.get_by_level(StorageLevel.L1_RAM.value)
        
        # Sort by importance (ascending) - least important first
        l1_entries.sort(key=lambda e: e.importance)
        
        return l1_entries[:count]

    def get_l2_candidates_for_archive(self, count: int = 10) -> List[IndexEntry]:
        """
        Get L2 nodes that are candidates for archival to L3.
        
        Selects based on importance and last access time.
        """
        l2_entries = self.get_by_level(StorageLevel.L2_WARM.value)
        
        # Sort by importance (ascending) - least important first
        l2_entries.sort(key=lambda e: e.importance)
        
        return l2_entries[:count]

    def update_location(self, node_id: str, file_path: str, byte_offset: int, level: int) -> None:
        """Update the location of a node after compaction."""
        if node_id in self.entries:
            entry = self.entries[node_id]
            
            # Update level index
            if entry.storage_level in self.level_index:
                self.level_index[entry.storage_level].discard(node_id)
            
            entry.storage_level = level
            entry.file_path = file_path
            entry.file_offset = byte_offset
            
            if level in self.level_index:
                self.level_index[level].add(node_id)

    def get_all_tags(self) -> List[str]:
        """Get all tags in the index."""
        return list(self.tag_index.keys())

    def get_all_types(self) -> List[str]:
        """Get all node types in the index."""
        return list(self.type_index.keys())

    def get_related_nodes(self, node_id: str, max_related: int = 5) -> List[IndexEntry]:
        """
        Get nodes related to a given node based on shared tags.
        
        Useful for graph traversal and context assembly.
        """
        if node_id not in self.entries:
            return []
        
        entry = self.entries[node_id]
        related_ids: Set[str] = set()
        
        # Find nodes with shared tags
        for tag in entry.tags:
            if tag in self.tag_index:
                related_ids.update(self.tag_index[tag])
        
        # Remove self
        related_ids.discard(node_id)
        
        # Get entries and sort by importance
        related = [self.entries[nid] for nid in related_ids if nid in self.entries]
        related.sort(key=lambda e: e.importance, reverse=True)
        
        return related[:max_related]

    def save(self) -> None:
        """Persist index to disk."""
        if not self.persistence_path:
            return
        
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        
        data = {
            "entries": {nid: entry.to_dict() for nid, entry in self.entries.items()},
            "tag_index": {tag: list(ids) for tag, ids in self.tag_index.items()},
            "type_index": {t: list(ids) for t, ids in self.type_index.items()},
            "level_index": {level: list(ids) for level, ids in self.level_index.items()},
            "saved_at": datetime.utcnow().isoformat()
        }
        
        with open(self.persistence_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        """Load index from disk."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return
        
        try:
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restore entries
            self.entries = {
                nid: IndexEntry.from_dict(entry_data)
                for nid, entry_data in data.get("entries", {}).items()
            }
            
            # Restore tag index
            self.tag_index = {
                tag: set(ids) for tag, ids in data.get("tag_index", {}).items()
            }
            
            # Restore type index
            self.type_index = {
                t: set(ids) for t, ids in data.get("type_index", {}).items()
            }
            
            # Restore level index
            self.level_index = {
                int(level): set(ids)
                for level, ids in data.get("level_index", {}).items()
            }
            
        except Exception as e:
            print(f"Error loading index: {e}")

    def clear(self) -> None:
        """Clear all index data."""
        self.entries.clear()
        self.tag_index.clear()
        self.type_index.clear()
        self.level_index = {
            StorageLevel.L1_RAM.value: set(),
            StorageLevel.L2_WARM.value: set(),
            StorageLevel.L3_COLD.value: set()
        }

    @property
    def size(self) -> int:
        """Get number of entries in the index."""
        return len(self.entries)

    def get_stats(self) -> Dict:
        """Get statistics about the index."""
        return {
            "total_entries": len(self.entries),
            "unique_tags": len(self.tag_index),
            "unique_types": len(self.type_index),
            "l1_nodes": len(self.level_index.get(StorageLevel.L1_RAM.value, set())),
            "l2_nodes": len(self.level_index.get(StorageLevel.L2_WARM.value, set())),
            "l3_nodes": len(self.level_index.get(StorageLevel.L3_COLD.value, set()))
        }