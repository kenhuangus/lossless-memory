"""
Memory Searcher - Provides search capabilities across all memory levels.

This module implements various search strategies to find relevant knowledge
nodes regardless of their storage level.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from ..knowledge_graph import KnowledgeNode, KnowledgeGraph, StorageLevel
from ..storage.ram_store import RAMStore
from ..storage.file_store import FileStore
from ..storage.index_manager import IndexManager, IndexEntry
from ..compaction.compactor import Compactor

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""
    node_id: str
    node: Optional[KnowledgeNode] = None
    relevance_score: float = 0.0
    storage_level: int = StorageLevel.L1_RAM.value
    summary: str = ""
    was_promoted: bool = False


class MemorySearcher:
    """
    Searches across all memory levels for relevant knowledge.
    
    Search Strategies:
    1. Index search - Fast search across summaries (all levels)
    2. Content search - Full-text search (slower, file-based)
    3. Graph traversal - Find connected nodes
    4. Tag search - Search by tags
    5. Type search - Search by node type
    """
    
    def __init__(
        self,
        ram_store: RAMStore,
        l2_store: FileStore,
        l3_store: FileStore,
        index_manager: IndexManager,
        compactor: Compactor,
        graph: KnowledgeGraph
    ):
        """
        Initialize memory searcher.
        
        Args:
            ram_store: L1 RAM storage
            l2_store: L2 warm file storage
            l3_store: L3 cold file storage
            index_manager: Global index manager
            compactor: Compactor for node promotion
            graph: Knowledge graph
        """
        self.ram_store = ram_store
        self.l2_store = l2_store
        self.l3_store = l3_store
        self.index_manager = index_manager
        self.compactor = compactor
        self.graph = graph

    def search(
        self,
        query: str,
        limit: int = 10,
        load_full: bool = False,
        auto_promote: bool = True
    ) -> List[SearchResult]:
        """
        Search for nodes matching query.
        
        Args:
            query: Search query string
            limit: Maximum results to return
            load_full: If True, load full node content
            auto_promote: If True, promote found nodes to L1
            
        Returns:
            List of SearchResult objects
        """
        # Search index first (fast)
        index_results = self.index_manager.search(query, limit=limit * 2)
        
        results = []
        seen_ids = set()
        
        for entry in index_results:
            if entry.node_id in seen_ids:
                continue
            seen_ids.add(entry.node_id)
            
            result = SearchResult(
                node_id=entry.node_id,
                relevance_score=entry.importance,
                storage_level=entry.storage_level,
                summary=entry.summary
            )
            
            # Load full node if requested
            if load_full or auto_promote:
                node = self._load_node(entry)
                if node:
                    result.node = node
                    
                    if auto_promote and entry.storage_level != StorageLevel.L1_RAM.value:
                        self.compactor.promote_to_l1(entry.node_id)
                        result.was_promoted = True
            
            results.append(result)
            
            if len(results) >= limit:
                break
        
        return results

    def search_by_tag(
        self,
        tag: str,
        limit: int = 10,
        load_full: bool = False
    ) -> List[SearchResult]:
        """Search for nodes with a specific tag."""
        entries = self.index_manager.search_by_tag(tag)
        
        results = []
        for entry in entries[:limit]:
            result = SearchResult(
                node_id=entry.node_id,
                relevance_score=entry.importance,
                storage_level=entry.storage_level,
                summary=entry.summary
            )
            
            if load_full:
                result.node = self._load_node(entry)
            
            results.append(result)
        
        return results

    def search_by_type(
        self,
        node_type: str,
        limit: int = 10,
        load_full: bool = False
    ) -> List[SearchResult]:
        """Search for nodes of a specific type."""
        entries = self.index_manager.search_by_type(node_type)
        
        # Sort by importance
        entries.sort(key=lambda e: e.importance, reverse=True)
        
        results = []
        for entry in entries[:limit]:
            result = SearchResult(
                node_id=entry.node_id,
                relevance_score=entry.importance,
                storage_level=entry.storage_level,
                summary=entry.summary
            )
            
            if load_full:
                result.node = self._load_node(entry)
            
            results.append(result)
        
        return results

    def get_related(
        self,
        node_id: str,
        limit: int = 5,
        load_full: bool = False
    ) -> List[SearchResult]:
        """
        Get nodes related to a given node.
        
        Uses both graph edges and tag similarity.
        """
        results = []
        seen_ids = {node_id}
        
        # Get graph neighbors
        neighbors = self.graph.get_neighbors(node_id)
        for neighbor_id in neighbors:
            if neighbor_id in seen_ids:
                continue
            seen_ids.add(neighbor_id)
            
            entry = self.index_manager.get_entry(neighbor_id)
            if entry:
                result = SearchResult(
                    node_id=neighbor_id,
                    relevance_score=entry.importance,
                    storage_level=entry.storage_level,
                    summary=entry.summary
                )
                
                if load_full:
                    result.node = self._load_node(entry)
                
                results.append(result)
        
        # Get related by tags
        tag_related = self.index_manager.get_related_nodes(node_id, max_related=limit)
        for entry in tag_related:
            if entry.node_id in seen_ids:
                continue
            seen_ids.add(entry.node_id)
            
            result = SearchResult(
                node_id=entry.node_id,
                relevance_score=entry.importance * 0.8,  # Lower score for tag-based
                storage_level=entry.storage_level,
                summary=entry.summary
            )
            
            if load_full:
                result.node = self._load_node(entry)
            
            results.append(result)
        
        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        return results[:limit]

    def get_recent(
        self,
        limit: int = 10,
        load_full: bool = False
    ) -> List[SearchResult]:
        """Get recently accessed nodes."""
        all_entries = list(self.index_manager.entries.values())
        
        # Sort by last accessed
        all_entries.sort(key=lambda e: e.last_accessed, reverse=True)
        
        results = []
        for entry in all_entries[:limit]:
            result = SearchResult(
                node_id=entry.node_id,
                relevance_score=entry.importance,
                storage_level=entry.storage_level,
                summary=entry.summary
            )
            
            if load_full:
                result.node = self._load_node(entry)
            
            results.append(result)
        
        return results

    def get_important(
        self,
        limit: int = 10,
        load_full: bool = False
    ) -> List[SearchResult]:
        """Get most important nodes."""
        all_entries = list(self.index_manager.entries.values())
        
        # Sort by importance
        all_entries.sort(key=lambda e: e.importance, reverse=True)
        
        results = []
        for entry in all_entries[:limit]:
            result = SearchResult(
                node_id=entry.node_id,
                relevance_score=entry.importance,
                storage_level=entry.storage_level,
                summary=entry.summary
            )
            
            if load_full:
                result.node = self._load_node(entry)
            
            results.append(result)
        
        return results

    def _load_node(self, entry: IndexEntry) -> Optional[KnowledgeNode]:
        """Load a full node from its storage level."""
        if entry.storage_level == StorageLevel.L1_RAM.value:
            return self.ram_store.get(entry.node_id)
        elif entry.storage_level == StorageLevel.L2_WARM.value:
            return self.l2_store.load_node(entry.node_id)
        elif entry.storage_level == StorageLevel.L3_COLD.value:
            return self.l3_store.load_node(entry.node_id)
        return None

    def get_context_window(
        self,
        query: str,
        max_tokens: int = 2000,
        include_related: bool = True
    ) -> Dict:
        """
        Get a context window of relevant nodes for a query.
        
        This is designed for LLM context assembly.
        
        Returns:
            Dict with 'nodes', 'summaries', and 'token_estimate'
        """
        # Get primary results
        results = self.search(query, limit=5, load_full=True, auto_promote=True)
        
        all_nodes = []
        all_summaries = []
        seen_ids = set()
        
        # Add primary results
        for result in results:
            if result.node and result.node_id not in seen_ids:
                all_nodes.append(result.node)
                all_summaries.append(result.summary)
                seen_ids.add(result.node_id)
        
        # Add related nodes if requested
        if include_related and results:
            for result in results[:3]:  # Top 3 primary results
                related = self.get_related(result.node_id, limit=3, load_full=True)
                for rel_result in related:
                    if rel_result.node and rel_result.node_id not in seen_ids:
                        all_nodes.append(rel_result.node)
                        all_summaries.append(rel_result.summary)
                        seen_ids.add(rel_result.node_id)
        
        # Estimate tokens (rough: 4 chars per token)
        token_estimate = sum(
            len(node.content) // 4 for node in all_nodes
        )
        
        return {
            "nodes": all_nodes,
            "summaries": all_summaries,
            "token_estimate": token_estimate,
            "node_count": len(all_nodes)
        }