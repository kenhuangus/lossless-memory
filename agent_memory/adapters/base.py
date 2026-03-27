"""
Base Adapter - Shared logic for all framework adapters.

All framework-specific adapters inherit from this class to get
consistent lossless memory behavior across every framework.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType, RelationType
from ..level_config import MemoryConfig, get_preset, create_assistant_config

logger = logging.getLogger(__name__)


class StorePattern(Enum):
    """Built-in patterns for what to store."""
    ALL_MESSAGES = "all_messages"
    KEYWORDS_ONLY = "keywords_only"
    DECISIONS_AND_FACTS = "decisions_and_facts"
    EVERYTHING = "everything"


@dataclass
class AdapterConfig:
    """
    Configuration for framework adapters.
    
    Controls what gets stored and how memory is injected.
    """
    # Storage patterns
    store_on_user: bool = True
    store_on_assistant: bool = False
    store_on_tool: bool = True
    store_on_error: bool = True
    
    # Keywords that trigger storage
    store_keywords: List[str] = field(default_factory=lambda: [
        "remember", "important", "decided", "choose", "learned",
        "mistake", "error", "solution", "fix", "key", "fact",
        "preference", "always", "never", "must", "critical"
    ])
    
    # Importance defaults
    default_importance: float = 0.5
    decision_importance: float = 0.8
    error_importance: float = 0.9
    fact_importance: float = 0.6
    
    # Context injection
    inject_context: bool = True
    max_context_tokens: int = 1500
    context_position: str = "system"
    
    # Auto-classification
    auto_detect_type: bool = True


class BaseMemoryAdapter:
    """
    Base class for all framework memory adapters.
    
    Provides shared logic for storing, retrieving, and injecting
    lossless memory into any agentic AI framework.
    
    Subclasses only need to implement framework-specific hooks.
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        config: Optional[AdapterConfig] = None,
        base_path: str = "./agent_memory_data",
        memory_preset: Optional[str] = None,
        memory_config: Optional[MemoryConfig] = None
    ):
        """
        Initialize the adapter.
        
        Args:
            memory: Existing MemoryManager instance
            config: Adapter-specific configuration
            base_path: Path for memory storage
            memory_preset: Preset name for memory levels
            memory_config: Custom memory configuration
        """
        if memory is not None:
            self.memory = memory
        else:
            if memory_config is not None:
                self.memory = MemoryManager(base_path=base_path, config=memory_config)
            elif memory_preset is not None:
                self.memory = MemoryManager(base_path=base_path, memory_preset=memory_preset)
            else:
                self.memory = MemoryManager(base_path=base_path)
        
        self.config = config or AdapterConfig()
        
        self.stats = {
            "messages_processed": 0,
            "items_stored": 0,
            "context_injections": 0
        }
        
        logger.info(f"BaseMemoryAdapter initialized")
    
    def before_turn(self, message: str, **kwargs) -> Optional[str]:
        """
        Called before the LLM processes a message.
        
        Retrieves relevant memory context to inject into the prompt.
        """
        if not self.config.inject_context:
            return None
        
        try:
            result = self.memory.get_context(
                query=message,
                max_tokens=self.config.max_context_tokens
            )
            
            if result["node_count"] > 0:
                self.stats["context_injections"] += 1
                return result["context"]
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting context: {e}")
            return None
    
    def after_turn(
        self,
        message: str,
        response: str,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> None:
        """
        Called after the LLM generates a response.
        
        Extracts and stores important information from the interaction.
        """
        self.stats["messages_processed"] += 1
        
        try:
            if self.config.store_on_user and self._should_store(message):
                self._store_message(message, "user", metadata)
            
            if self.config.store_on_assistant and self._should_store(response):
                self._store_message(response, "assistant", metadata)
                
        except Exception as e:
            logger.warning(f"Error storing turn: {e}")
    
    def on_tool_call(
        self,
        tool_name: str,
        tool_input: Dict,
        tool_output: Any,
        **kwargs
    ) -> None:
        """
        Called when a tool is used.
        """
        if not self.config.store_on_tool:
            return
        
        try:
            content = f"Used tool '{tool_name}' with input: {str(tool_input)[:500]}. Result: {str(tool_output)[:500]}"
            
            self.memory.remember(
                content=content,
                node_type=NodeType.EXPERIENCE.value,
                tags=["tool_usage", tool_name],
                importance=self.config.default_importance
            )
            self.stats["items_stored"] += 1
            
        except Exception as e:
            logger.warning(f"Error storing tool call: {e}")
    
    def on_error(self, error: Exception, context: str = "", **kwargs) -> None:
        """
        Called when an error occurs.
        """
        if not self.config.store_on_error:
            return
        
        try:
            self.memory.remember(
                content=f"Error: {str(error)}. Context: {context}",
                node_type=NodeType.ERROR.value,
                tags=["error", type(error).__name__],
                importance=self.config.error_importance
            )
            self.stats["items_stored"] += 1
            
        except Exception as e:
            logger.warning(f"Error storing error: {e}")
    
    def remember(
        self,
        content: str,
        node_type: str = NodeType.FACT.value,
        tags: List[str] = None,
        importance: float = None,
        **kwargs
    ) -> str:
        """Store knowledge directly."""
        importance = importance or self.config.default_importance
        node_id = self.memory.remember(
            content=content,
            node_type=node_type,
            tags=tags or [],
            importance=importance,
            **kwargs
        )
        self.stats["items_stored"] += 1
        return node_id
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory directly."""
        return self.memory.recall(query=query, limit=limit)
    
    def get(self, node_id: str):
        """Get a specific node by ID."""
        return self.memory.get(node_id)
    
    def get_context(self, query: str, max_tokens: int = 2000) -> Dict:
        """Get formatted context for LLM."""
        return self.memory.get_context(query=query, max_tokens=max_tokens)
    
    def save(self) -> None:
        """Save memory to disk."""
        self.memory.save()
    
    def get_stats(self) -> Dict:
        """Get combined statistics."""
        return {
            "adapter": self.stats,
            "memory": self.memory.get_stats()
        }
    
    def get_memory_summary(self) -> str:
        """Get human-readable memory summary."""
        return self.memory.get_memory_summary()
    
    def clear(self) -> None:
        """Clear all memory."""
        self.memory.clear()
        self.stats = {
            "messages_processed": 0,
            "items_stored": 0,
            "context_injections": 0
        }
    
    def _should_store(self, text: str) -> bool:
        """Determine if text should be stored based on config."""
        if not text or len(text.strip()) < 10:
            return False
        
        text_lower = text.lower()
        
        for keyword in self.config.store_keywords:
            if keyword.lower() in text_lower:
                return True
        
        return len(text) > 100
    
    def _store_message(
        self,
        content: str,
        role: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Store a message with auto-classification."""
        node_type = NodeType.FACT.value
        importance = self.config.default_importance
        tags = [role, "auto_stored"]
        
        if self.config.auto_detect_type:
            content_lower = content.lower()
            
            if any(w in content_lower for w in ["decided", "choose", "will", "going to", "plan to"]):
                node_type = NodeType.DECISION.value
                importance = self.config.decision_importance
                tags.append("decision")
            elif any(w in content_lower for w in ["error", "failed", "wrong", "mistake", "bug"]):
                node_type = NodeType.ERROR.value
                importance = self.config.error_importance
                tags.append("error")
            elif any(w in content_lower for w in ["solution", "fix", "solved", "resolved"]):
                node_type = NodeType.SOLUTION.value
                importance = self.config.decision_importance
                tags.append("solution")
            elif any(w in content_lower for w in ["is", "are", "means", "defined as", "refers to"]):
                node_type = NodeType.FACT.value
                importance = self.config.fact_importance
                tags.append("fact")
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    tags.append(f"{key}:{value}")
        
        self.memory.remember(
            content=content[:2000],
            node_type=node_type,
            tags=tags,
            importance=importance
        )
        self.stats["items_stored"] += 1
    
    def _format_context_for_injection(self, context: str) -> str:
        """Format memory context for injection into prompts."""
        if not context:
            return ""
        return f"""
## Relevant Knowledge from Memory:
{context}
---
"""