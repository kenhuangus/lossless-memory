"""
Memory-Enhanced Agent for Google ADK.

This module provides an agent class that automatically integrates
long-term memory capabilities with Google ADK agents.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class MemoryEnhancedAgent:
    """
    Wrapper that adds long-term memory to any Google ADK agent.
    
    This class wraps an ADK agent and automatically:
    - Stores important information from conversations
    - Retrieves relevant context before each response
    - Manages memory compaction and maintenance
    
    Usage:
        from google.adk.agents import LlmAgent
        from agent_memory.adk import MemoryEnhancedAgent
        
        # Create base ADK agent
        base_agent = LlmAgent(...)
        
        # Wrap with memory
        agent = MemoryEnhancedAgent(
            agent=base_agent,
            memory_path="./my_agent_memory"
        )
        
        # Use normally - memory is automatic
        response = await agent.run("What did we discuss yesterday?")
    """
    
    def __init__(
        self,
        agent: Any,
        memory_path: str = "./agent_memory",
        memory_capacity: int = 100,
        auto_store: bool = True,
        auto_retrieve: bool = True,
        store_callback: Optional[Callable] = None
    ):
        """
        Initialize memory-enhanced agent.
        
        Args:
            agent: The base ADK agent to wrap
            memory_path: Path for memory storage
            memory_capacity: L1 memory capacity
            auto_store: Automatically store important information
            auto_retrieve: Automatically retrieve relevant context
            store_callback: Optional callback to decide what to store
        """
        self.agent = agent
        self.auto_store = auto_store
        self.auto_retrieve = auto_retrieve
        self.store_callback = store_callback
        
        # Initialize memory system
        self.memory = MemoryManager(
            base_path=memory_path,
            l1_capacity=memory_capacity
        )
        
        # Track conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"MemoryEnhancedAgent initialized at {memory_path}")
    
    async def run(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Run the agent with automatic memory management.
        
        Args:
            message: User message
            context: Optional context
            **kwargs: Additional arguments for the agent
            
        Returns:
            Agent response
        """
        # Step 1: Retrieve relevant context from memory
        memory_context = ""
        if self.auto_retrieve:
            memory_context = self._retrieve_context(message)
        
        # Step 2: Enhance the message with memory context
        enhanced_message = self._enhance_message(message, memory_context)
        
        # Step 3: Run the base agent
        try:
            response = await self.agent.run(
                message=enhanced_message,
                context=context,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            raise
        
        # Step 4: Store important information from the interaction
        if self.auto_store:
            self._store_interaction(message, response)
        
        # Step 5: Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": str(response),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return response
    
    def _retrieve_context(self, message: str) -> str:
        """
        Retrieve relevant context from memory.
        
        Args:
            message: User message
            
        Returns:
            Formatted context string
        """
        try:
            context_result = self.memory.get_context(
                query=message,
                max_tokens=1500,
                include_summaries=True,
                include_full_content=False
            )
            
            if context_result["node_count"] > 0:
                return f"""
## Relevant Knowledge from Memory:
{context_result["context"]}
---
"""
            return ""
        except Exception as e:
            logger.warning(f"Failed to retrieve context: {e}")
            return ""
    
    def _enhance_message(self, message: str, memory_context: str) -> str:
        """
        Enhance the user message with memory context.
        
        Args:
            message: Original user message
            memory_context: Context from memory
            
        Returns:
            Enhanced message
        """
        if not memory_context:
            return message
        
        return f"""{memory_context}

## User Question:
{message}

Please use the relevant knowledge from memory above to inform your response when applicable.
"""
    
    def _store_interaction(self, message: str, response: Any) -> None:
        """
        Store important information from the interaction.
        
        Args:
            message: User message
            response: Agent response
        """
        try:
            # Determine what to store based on content
            if self.store_callback:
                # Use custom callback
                store_decision = self.store_callback(message, response)
                if store_decision:
                    self.memory.remember(**store_decision)
            else:
                # Default storage logic
                self._default_store(message, response)
        except Exception as e:
            logger.warning(f"Failed to store interaction: {e}")
    
    def _default_store(self, message: str, response: Any) -> None:
        """
        Default logic for storing interactions.
        
        Stores:
        - User questions that seem important
        - Agent decisions and conclusions
        - Facts and information shared
        """
        response_str = str(response)
        
        # Store user questions that seem like decisions or important info
        decision_keywords = ["decide", "choose", "prefer", "want", "need", "should", "will"]
        if any(keyword in message.lower() for keyword in decision_keywords):
            self.memory.remember(
                content=f"User stated: {message}",
                node_type=NodeType.DECISION.value,
                tags=["user_input", "decision"],
                importance=0.7
            )
        
        # Store agent responses that contain conclusions or facts
        conclusion_keywords = ["therefore", "conclusion", "recommend", "suggest", "answer is", "solution"]
        if any(keyword in response_str.lower() for keyword in conclusion_keywords):
            # Extract the conclusion (first 500 chars)
            conclusion = response_str[:500]
            self.memory.remember(
                content=f"Agent conclusion: {conclusion}",
                node_type=NodeType.SOLUTION.value,
                tags=["agent_output", "conclusion"],
                importance=0.6
            )
        
        # Store factual information
        fact_indicators = ["is", "are", "was", "were", "has", "have", "means", "refers to"]
        if any(indicator in message.lower() for indicator in fact_indicators):
            if len(message) > 50:  # Only store substantial facts
                self.memory.remember(
                    content=message,
                    node_type=NodeType.FACT.value,
                    tags=["user_fact"],
                    importance=0.5
                )
    
    def remember_manually(
        self,
        content: str,
        node_type: str = "fact",
        tags: List[str] = None,
        importance: float = 0.5
    ) -> str:
        """
        Manually store knowledge in memory.
        
        Args:
            content: Knowledge content
            node_type: Type of knowledge
            tags: Tags for categorization
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
        Search memory for knowledge.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of results
        """
        return self.memory.recall(query=query, limit=limit)
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        return self.memory.get_stats()
    
    def save_memory(self) -> None:
        """Save memory to disk."""
        self.memory.save()
    
    def clear_memory(self) -> None:
        """Clear all memory."""
        self.memory.clear()
        self.conversation_history.clear()
    
    def export_memory(self, path: str) -> None:
        """
        Export memory to markdown file.
        
        Args:
            path: Output file path
        """
        self.memory.export_markdown(path)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()