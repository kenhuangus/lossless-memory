"""
Cline Memory Adapter.

Provides lossless memory for Cline code agents.

Usage:
    from agent_memory.adapters import ClineMemory
    
    memory = ClineMemory(base_path="./my_memory")
    
    # Store code context
    memory.remember_code("def hello(): print('Hello')")
    
    # Retrieve relevant code
    results = memory.recall_code("hello function")
"""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class ClineMemory:
    """
    Memory adapter for Cline code agents.
    
    Specialized for storing and retrieving code snippets,
    file contexts, and coding decisions.
    
    Usage:
        from agent_memory.adapters import ClineMemory
        
        memory = ClineMemory(base_path="./cline_memory")
        
        # Store code context
        memory.remember_code(
            code="def authenticate(user, password): ...",
            file_path="auth.py",
            language="python"
        )
        
        # Retrieve relevant code
        results = memory.recall_code("authentication function")
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./cline_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize Cline memory."""
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset or "assistant"
        )
        
        logger.info("ClineMemory initialized")
    
    def remember_code(
        self,
        code: str,
        file_path: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
        importance: float = 0.6
    ) -> str:
        """
        Store a code snippet.
        
        Args:
            code: The code content
            file_path: File path where code is from
            language: Programming language
            description: Description of what the code does
            importance: Importance score
            
        Returns:
            Node ID
        """
        tags = ["code"]
        if language:
            tags.append(f"lang:{language}")
        if file_path:
            tags.append(f"file:{file_path}")
        
        content = code
        if description:
            content = f"{description}\n\n```{language or ''}\n{code}\n```"
        
        return self._adapter.remember(
            content=content,
            node_type="fact",
            tags=tags,
            importance=importance,
            metadata={
                "file_path": file_path,
                "language": language,
                "description": description
            }
        )
    
    def remember_decision(
        self,
        decision: str,
        rationale: str,
        alternatives: Optional[List[str]] = None,
        importance: float = 0.8
    ) -> str:
        """
        Store a coding decision.
        
        Args:
            decision: The decision made
            rationale: Why this decision was made
            alternatives: Alternative approaches considered
            importance: Importance score
            
        Returns:
            Node ID
        """
        content = f"Decision: {decision}\nRationale: {rationale}"
        if alternatives:
            content += f"\nAlternatives considered: {', '.join(alternatives)}"
        
        return self._adapter.remember(
            content=content,
            node_type="decision",
            tags=["decision", "coding"],
            importance=importance
        )
    
    def remember_error(
        self,
        error: str,
        solution: str,
        context: Optional[str] = None,
        importance: float = 0.9
    ) -> str:
        """
        Store an error and its solution.
        
        Args:
            error: The error encountered
            solution: How it was solved
            context: Additional context
            importance: Importance score
            
        Returns:
            Node ID
        """
        content = f"Error: {error}\nSolution: {solution}"
        if context:
            content += f"\nContext: {context}"
        
        return self._adapter.remember(
            content=content,
            node_type="solution",
            tags=["error", "solution", "coding"],
            importance=importance
        )
    
    def remember_file(
        self,
        file_path: str,
        content: str,
        language: Optional[str] = None,
        importance: float = 0.5
    ) -> str:
        """
        Store file content.
        
        Args:
            file_path: Path to the file
            content: File content
            language: Programming language
            importance: Importance score
            
        Returns:
            Node ID
        """
        tags = ["file", f"path:{file_path}"]
        if language:
            tags.append(f"lang:{language}")
        
        return self._adapter.remember(
            content=content,
            node_type="fact",
            tags=tags,
            importance=importance,
            metadata={"file_path": file_path, "language": language}
        )
    
    def recall_code(
        self,
        query: str,
        language: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for relevant code.
        
        Args:
            query: Search query
            language: Filter by language
            limit: Max results
            
        Returns:
            List of code results
        """
        search_query = query
        if language:
            search_query += f" lang:{language}"
        
        return self._adapter.recall(query=search_query, limit=limit)
    
    def recall_decisions(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant decisions."""
        return self._adapter.recall(query=f"decision {query}", limit=limit)
    
    def recall_errors(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant errors/solutions."""
        return self._adapter.recall(query=f"error solution {query}", limit=limit)
    
    def get_file_context(
        self,
        file_path: str,
        max_tokens: int = 2000
    ) -> str:
        """Get context for a specific file."""
        result = self._adapter.get_context(
            query=f"file:{file_path}",
            max_tokens=max_tokens
        )
        return result.get("context", "")
    
    def remember(self, content: str, **kwargs) -> str:
        """Store knowledge directly."""
        return self._adapter.remember(content, **kwargs)
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory directly."""
        return self._adapter.recall(query, limit)
    
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get context for a query."""
        result = self._adapter.get_context(query=query, max_tokens=max_tokens)
        return result.get("context", "")
    
    def before_turn(self, message: str) -> Optional[str]:
        """Get context before processing a message."""
        return self._adapter.before_turn(message)
    
    def after_turn(self, message: str, response: str, **kwargs) -> None:
        """Store after processing a message."""
        self._adapter.after_turn(message, response, **kwargs)
    
    def save(self) -> None:
        """Save memory to disk."""
        self._adapter.save()
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self._adapter.get_stats()