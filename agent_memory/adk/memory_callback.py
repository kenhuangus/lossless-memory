"""
Memory Callback for Google ADK.

This module provides callback functionality for automatic memory
management with Google ADK agents.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType, RelationType

logger = logging.getLogger(__name__)


class MemoryCallback:
    """
    Callback system for automatic memory management with ADK agents.
    
    This class provides hooks that can be attached to ADK agent events
    to automatically store and retrieve knowledge.
    
    Usage:
        from google.adk.agents import LlmAgent
        from agent_memory.adk import MemoryCallback
        
        # Create memory callback
        memory_cb = MemoryCallback(memory_path="./my_memory")
        
        # Create agent with callback
        agent = LlmAgent(
            ...,
            callbacks=[memory_cb]
        )
        
        # Or use with agent runner
        runner = AgentRunner(agent=agent, callbacks=[memory_cb])
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        memory_path: str = "./agent_memory",
        memory_capacity: int = 100,
        auto_store_on_response: bool = True,
        auto_retrieve_on_message: bool = True,
        store_patterns: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize memory callback.
        
        Args:
            memory: Existing MemoryManager instance (optional)
            memory_path: Path for memory storage (if memory not provided)
            memory_capacity: L1 memory capacity
            auto_store_on_response: Auto-store agent responses
            auto_retrieve_on_message: Auto-retrieve context for messages
            store_patterns: Custom patterns for what to store
        """
        # Initialize or use existing memory
        if memory:
            self.memory = memory
        else:
            self.memory = MemoryManager(
                base_path=memory_path,
                l1_capacity=memory_capacity
            )
        
        self.auto_store_on_response = auto_store_on_response
        self.auto_retrieve_on_message = auto_retrieve_on_message
        
        # Default storage patterns
        self.store_patterns = store_patterns or {
            "decisions": {
                "keywords": ["decide", "choose", "prefer", "will", "should", "must"],
                "node_type": NodeType.DECISION.value,
                "importance": 0.8
            },
            "errors": {
                "keywords": ["error", "failed", "mistake", "wrong", "bug", "issue"],
                "node_type": NodeType.ERROR.value,
                "importance": 0.9
            },
            "solutions": {
                "keywords": ["solution", "fix", "solve", "resolved", "answer"],
                "node_type": NodeType.SOLUTION.value,
                "importance": 0.85
            },
            "facts": {
                "keywords": ["is", "are", "means", "refers to", "defined as"],
                "node_type": NodeType.FACT.value,
                "importance": 0.5,
                "min_length": 50
            }
        }
        
        # Track context for relationship creation
        self.last_stored_node_id: Optional[str] = None
        
        logger.info(f"MemoryCallback initialized at {memory_path}")
    
    def on_message(self, message: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Called when a user message is received.
        
        Args:
            message: User message
            **kwargs: Additional context
            
        Returns:
            Context dict to inject, or None
        """
        if not self.auto_retrieve_on_message:
            return None
        
        try:
            # Retrieve relevant context
            context_result = self.memory.get_context(
                query=message,
                max_tokens=1500,
                include_summaries=True,
                include_full_content=False
            )
            
            if context_result["node_count"] > 0:
                return {
                    "memory_context": context_result["context"],
                    "memory_node_count": context_result["node_count"]
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve context: {e}")
            return None
    
    def on_response(self, response: str, message: str = "", **kwargs) -> None:
        """
        Called when an agent response is generated.
        
        Args:
            response: Agent response
            message: Original user message
            **kwargs: Additional context
        """
        if not self.auto_store_on_response:
            return
        
        try:
            # Store based on patterns
            self._store_by_patterns(message, response)
            
        except Exception as e:
            logger.warning(f"Failed to store response: {e}")
    
    def on_tool_call(self, tool_name: str, tool_input: Dict, tool_output: Dict, **kwargs) -> None:
        """
        Called when a tool is used.
        
        Args:
            tool_name: Name of the tool
            tool_input: Tool input
            tool_output: Tool output
            **kwargs: Additional context
        """
        try:
            # Store tool usage as experience
            self.memory.remember(
                content=f"Used tool '{tool_name}' with input: {tool_input}. Result: {tool_output}",
                node_type=NodeType.EXPERIENCE.value,
                tags=["tool_usage", tool_name],
                importance=0.6
            )
        except Exception as e:
            logger.warning(f"Failed to store tool usage: {e}")
    
    def on_error(self, error: Exception, context: str = "", **kwargs) -> None:
        """
        Called when an error occurs.
        
        Args:
            error: The exception
            context: Error context
            **kwargs: Additional context
        """
        try:
            self.memory.remember(
                content=f"Error: {str(error)}. Context: {context}",
                node_type=NodeType.ERROR.value,
                tags=["error", type(error).__name__],
                importance=0.9
            )
        except Exception as e:
            logger.warning(f"Failed to store error: {e}")
    
    def _store_by_patterns(self, message: str, response: str) -> None:
        """
        Store content based on pattern matching.
        
        Args:
            message: User message
            response: Agent response
        """
        combined_text = f"{message} {response}".lower()
        
        for pattern_name, pattern_config in self.store_patterns.items():
            keywords = pattern_config.get("keywords", [])
            
            # Check if any keyword matches
            if any(keyword in combined_text for keyword in keywords):
                # Check minimum length if specified
                min_length = pattern_config.get("min_length", 0)
                if len(response) < min_length:
                    continue
                
                # Determine content to store
                if pattern_name == "decisions":
                    content = f"Decision made: {message}"
                elif pattern_name == "errors":
                    content = f"Error encountered: {response[:500]}"
                elif pattern_name == "solutions":
                    content = f"Solution found: {response[:500]}"
                else:
                    content = response[:500]
                
                # Store the knowledge
                node_id = self.memory.remember(
                    content=content,
                    node_type=pattern_config.get("node_type", NodeType.FACT.value),
                    tags=[pattern_name, "auto_stored"],
                    importance=pattern_config.get("importance", 0.5)
                )
                
                # Create relationship to previous node if exists
                if self.last_stored_node_id:
                    try:
                        self.memory.add_relation(
                            source_id=self.last_stored_node_id,
                            target_id=node_id,
                            relation_type=RelationType.PRECEDES.value
                        )
                    except Exception:
                        pass  # Ignore relationship errors
                
                self.last_stored_node_id = node_id
                
                # Only store for first matching pattern
                break
    
    def remember(
        self,
        content: str,
        node_type: str = "fact",
        tags: List[str] = None,
        importance: float = 0.5
    ) -> str:
        """
        Manually store knowledge.
        
        Args:
            content: Knowledge content
            node_type: Type of knowledge
            tags: Tags
            importance: Importance score
            
        Returns:
            Node ID
        """
        return self.memory.remember(
            content=content,
            node_type=node_type,
            tags=tags or [],
            importance=importance
        )
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search memory.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of results
        """
        return self.memory.recall(query=query, limit=limit)
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self.memory.get_stats()
    
    def save(self) -> None:
        """Save memory to disk."""
        self.memory.save()
    
    def clear(self) -> None:
        """Clear all memory."""
        self.memory.clear()
        self.last_stored_node_id = None