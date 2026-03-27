m"""
Memory Tool for Google ADK.

This tool allows ADK agents to interact with the Agent Memory System
for storing, retrieving, and searching knowledge.
"""

from typing import Any, Dict, List, Optional
from google.adk.tools import FunctionTool

from ..memory_manager import MemoryManager


def create_memory_tool(memory: MemoryManager) -> FunctionTool:
    """
    Create a memory tool for ADK agents.
    
    Args:
        memory: MemoryManager instance
        
    Returns:
        FunctionTool that can be used with ADK agents
    """
    
    def remember(
        content: str,
        node_type: str = "fact",
        tags: List[str] = None,
        importance: float = 0.5,
        summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store a piece of knowledge in long-term memory.
        
        Args:
            content: The knowledge content to remember
            node_type: Type (fact, decision, experience, error, solution, goal, plan)
            tags: Tags for categorization
            importance: Importance (0-1)
            summary: Optional summary
            
        Returns:
            Dict with success status and node_id
        """
        try:
            node_id = memory.remember(
                content=content,
                node_type=node_type,
                tags=tags or [],
                importance=importance,
                summary=summary
            )
            return {
                "success": True,
                "node_id": node_id,
                "message": f"Stored knowledge with ID: {node_id}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def recall(
        query: str,
        limit: int = 5,
        load_full: bool = False
    ) -> Dict[str, Any]:
        """
        Search for knowledge in long-term memory.
        
        Args:
            query: Search query
            limit: Maximum results to return
            load_full: Load full content
            
        Returns:
            Dict with results and count
        """
        try:
            results = memory.recall(
                query=query,
                limit=limit,
                load_full=load_full
            )
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_context(
        query: str,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Get formatted context for LLM based on a query.
        
        Args:
            query: Query or topic for context
            max_tokens: Maximum tokens
            
        Returns:
            Dict with context and metadata
        """
        try:
            context = memory.get_context(
                query=query,
                max_tokens=max_tokens
            )
            return {
                "success": True,
                "context": context["context"],
                "token_estimate": context["token_estimate"],
                "node_count": context["node_count"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_related(
        node_id: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Get knowledge related to a specific node.
        
        Args:
            node_id: Node ID to find related knowledge for
            limit: Maximum results
            
        Returns:
            Dict with related results
        """
        try:
            results = memory.get_related(
                node_id=node_id,
                limit=limit
            )
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_stats() -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Dict with stats and summary
        """
        try:
            stats = memory.get_stats()
            summary = memory.get_memory_summary()
            return {
                "success": True,
                "stats": stats,
                "summary": summary
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_by_tag(
        tag: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for knowledge by tag.
        
        Args:
            tag: Tag to search for
            limit: Maximum results
            
        Returns:
            Dict with results
        """
        try:
            results = memory.search_by_tag(
                tag=tag,
                limit=limit
            )
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def memory_tool(
        action: str,
        content: str = "",
        node_type: str = "fact",
        tags: List[str] = None,
        importance: float = 0.5,
        summary: Optional[str] = None,
        query: str = "",
        node_id: str = "",
        tag: str = "",
        limit: int = 5,
        max_tokens: int = 2000,
        load_full: bool = False
    ) -> Dict[str, Any]:
        """
        Long-term memory system for storing and retrieving knowledge.
        
        Actions:
        - remember: Store knowledge (params: content, node_type, tags, importance, summary)
        - recall: Search knowledge (params: query, limit, load_full)
        - get_context: Get LLM context (params: query, max_tokens)
        - get_related: Get related knowledge (params: node_id, limit)
        - get_stats: Get memory statistics
        - search_by_tag: Search by tag (params: tag, limit)
        """
        if action == "remember":
            return remember(content, node_type, tags, importance, summary)
        elif action == "recall":
            return recall(query, limit, load_full)
        elif action == "get_context":
            return get_context(query, max_tokens)
        elif action == "get_related":
            return get_related(node_id, limit)
        elif action == "get_stats":
            return get_stats()
        elif action == "search_by_tag":
            return search_by_tag(tag, limit)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    
    # Create the tool with the unified function
    tool = FunctionTool(func=memory_tool)
    tool.name = "memory"
    tool.description = "Long-term memory system. Actions: remember, recall, get_context, get_related, get_stats, search_by_tag"
    return tool
