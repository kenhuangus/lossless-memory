"""
Context Assembler - Assembles context for LLM prompts.

This module prepares knowledge from memory for use in LLM context windows,
optimizing for relevance and token efficiency.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..knowledge_graph import KnowledgeNode, StorageLevel
from ..storage.index_manager import IndexManager
from .searcher import MemorySearcher, SearchResult

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Assembles context from memory for LLM prompts.
    
    Features:
    - Selects most relevant nodes for a given query
    - Formats nodes for optimal LLM consumption
    - Manages token budget
    - Provides summaries and full content as needed
    """
    
    # Approximate tokens per character (conservative estimate)
    TOKENS_PER_CHAR = 0.25
    
    def __init__(
        self,
        searcher: MemorySearcher,
        index_manager: IndexManager,
        default_max_tokens: int = 4000
    ):
        """
        Initialize context assembler.
        
        Args:
            searcher: Memory searcher for finding relevant nodes
            index_manager: Global index manager
            default_max_tokens: Default token budget for context
        """
        self.searcher = searcher
        self.index_manager = index_manager
        self.default_max_tokens = default_max_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.TOKENS_PER_CHAR)

    def assemble_context(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        include_summaries: bool = True,
        include_full_content: bool = True,
        include_metadata: bool = False
    ) -> Dict:
        """
        Assemble context for a query.
        
        Args:
            query: The query or topic for context
            max_tokens: Maximum tokens to include
            include_summaries: Include node summaries
            include_full_content: Include full node content
            include_metadata: Include node metadata
            
        Returns:
            Dict with formatted context and metadata
        """
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        # Get relevant nodes
        results = self.searcher.search(
            query, 
            limit=20, 
            load_full=True, 
            auto_promote=True
        )
        
        # Assemble context within token budget
        context_parts = []
        used_tokens = 0
        included_nodes = []
        
        # First pass: add summaries (cheap on tokens)
        if include_summaries:
            for result in results:
                if result.summary:
                    summary_text = f"[{result.node_id[:8]}] {result.summary}"
                    tokens = self.estimate_tokens(summary_text)
                    
                    if used_tokens + tokens <= max_tokens * 0.3:
                        context_parts.append({
                            "type": "summary",
                            "node_id": result.node_id,
                            "content": summary_text,
                            "tokens": tokens
                        })
                        used_tokens += tokens
                        included_nodes.append(result.node_id)
        
        # Second pass: add full content for most relevant nodes
        if include_full_content:
            for result in results:
                if result.node and result.node_id not in included_nodes:
                    content_text = self._format_node(result.node, include_metadata)
                    tokens = self.estimate_tokens(content_text)
                    
                    if used_tokens + tokens <= max_tokens:
                        context_parts.append({
                            "type": "full",
                            "node_id": result.node_id,
                            "content": content_text,
                            "tokens": tokens
                        })
                        used_tokens += tokens
                        included_nodes.append(result.node_id)
        
        # Format final context
        formatted_context = self._format_context_parts(context_parts)
        
        return {
            "context": formatted_context,
            "token_estimate": used_tokens,
            "node_count": len(included_nodes),
            "node_ids": included_nodes,
            "query": query,
            "timestamp": datetime.utcnow().isoformat()
        }

    def assemble_from_nodes(
        self,
        nodes: List[KnowledgeNode],
        max_tokens: Optional[int] = None,
        include_metadata: bool = False
    ) -> Dict:
        """
        Assemble context from a specific list of nodes.
        
        Useful when you already know which nodes to include.
        """
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        context_parts = []
        used_tokens = 0
        included_nodes = []
        
        for node in nodes:
            content_text = self._format_node(node, include_metadata)
            tokens = self.estimate_tokens(content_text)
            
            if used_tokens + tokens <= max_tokens:
                context_parts.append({
                    "type": "full",
                    "node_id": node.id,
                    "content": content_text,
                    "tokens": tokens
                })
                used_tokens += tokens
                included_nodes.append(node.id)
        
        formatted_context = self._format_context_parts(context_parts)
        
        return {
            "context": formatted_context,
            "token_estimate": used_tokens,
            "node_count": len(included_nodes),
            "node_ids": included_nodes,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _format_node(
        self, 
        node: KnowledgeNode, 
        include_metadata: bool = False
    ) -> str:
        """Format a single node for context."""
        lines = []
        lines.append(f"## {node.node_type.upper()}: {node.id[:8]}")
        lines.append(f"Tags: {', '.join(node.tags) if node.tags else 'none'}")
        lines.append("")
        lines.append(node.content)
        
        if include_metadata and node.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in node.metadata.items():
                lines.append(f"  {key}: {value}")
        
        lines.append("")
        return "\n".join(lines)

    def _format_context_parts(self, parts: List[Dict]) -> str:
        """Format all context parts into a single string."""
        if not parts:
            return ""
        
        sections = []
        
        # Group by type
        summaries = [p for p in parts if p["type"] == "summary"]
        full_content = [p for p in parts if p["type"] == "full"]
        
        if summaries:
            sections.append("# Memory Summaries")
            sections.append("")
            for part in summaries:
                sections.append(f"- {part['content']}")
            sections.append("")
        
        if full_content:
            sections.append("# Detailed Knowledge")
            sections.append("")
            for part in full_content:
                sections.append(part['content'])
        
        return "\n".join(sections)

    def get_working_memory_context(self, max_tokens: int = 2000) -> Dict:
        """
        Get context of current working memory (L1 nodes).
        
        Useful for understanding what the agent is currently focused on.
        """
        l1_entries = self.index_manager.get_by_level(StorageLevel.L1_RAM.value)
        
        # Sort by importance
        l1_entries.sort(key=lambda e: e.importance, reverse=True)
        
        context_parts = []
        used_tokens = 0
        included_nodes = []
        
        for entry in l1_entries:
            node = self.searcher._load_node(entry)
            if node:
                content_text = self._format_node(node, include_metadata=False)
                tokens = self.estimate_tokens(content_text)
                
                if used_tokens + tokens <= max_tokens:
                    context_parts.append({
                        "type": "full",
                        "node_id": node.id,
                        "content": content_text,
                        "tokens": tokens
                    })
                    used_tokens += tokens
                    included_nodes.append(node.id)
        
        formatted_context = self._format_context_parts(context_parts)
        
        return {
            "context": formatted_context,
            "token_estimate": used_tokens,
            "node_count": len(included_nodes),
            "node_ids": included_nodes
        }

    def get_memory_summary(self) -> str:
        """Get a human-readable summary of memory state."""
        index_stats = self.index_manager.get_stats()
        
        lines = []
        lines.append("# Memory System Status")
        lines.append("")
        lines.append(f"Total nodes: {index_stats['total_entries']}")
        lines.append(f"L1 (RAM): {index_stats['l1_nodes']} nodes")
        lines.append(f"L2 (Warm): {index_stats['l2_nodes']} nodes")
        lines.append(f"L3 (Cold): {index_stats['l3_nodes']} nodes")
        lines.append(f"Unique tags: {index_stats['unique_tags']}")
        lines.append(f"Node types: {index_stats['unique_types']}")
        
        return "\n".join(lines)