"""
Task Memory Adapter.

Provides lossless memory for Task code agents.

Usage:
    from agent_memory.adapters import TaskMemory
    
    memory = TaskMemory(base_path="./my_memory")
    
    # Store task context
    memory.remember_task("Implement user authentication", status="in_progress")
    
    # Retrieve relevant tasks
    results = memory.recall_tasks("authentication")
"""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class TaskMemory:
    """
    Memory adapter for Task code agents.
    
    Specialized for storing and retrieving task context,
    task decisions, and task-related code.
    
    Usage:
        from agent_memory.adapters import TaskMemory
        
        memory = TaskMemory(base_path="./task_memory")
        
        # Store task context
        memory.remember_task(
            task="Implement user authentication",
            description="Add JWT-based auth to API",
            status="in_progress"
        )
        
        # Retrieve relevant tasks
        results = memory.recall_tasks("authentication")
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./task_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize Task memory."""
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset or "assistant"
        )
        
        logger.info("TaskMemory initialized")
    
    def remember_task(
        self,
        task: str,
        description: Optional[str] = None,
        status: str = "pending",
        priority: str = "medium",
        tags: Optional[List[str]] = None,
        importance: float = 0.6
    ) -> str:
        """
        Store a task.
        
        Args:
            task: Task name/title
            description: Task description
            status: Task status (pending, in_progress, completed, blocked)
            priority: Task priority (low, medium, high, critical)
            tags: Additional tags
            importance: Importance score
            
        Returns:
            Node ID
        """
        all_tags = ["task", f"status:{status}", f"priority:{priority}"]
        if tags:
            all_tags.extend(tags)
        
        content = f"Task: {task}"
        if description:
            content += f"\nDescription: {description}"
        content += f"\nStatus: {status}, Priority: {priority}"
        
        return self._adapter.remember(
            content=content,
            node_type="decision",
            tags=all_tags,
            importance=importance,
            metadata={
                "task": task,
                "status": status,
                "priority": priority,
                "description": description
            }
        )
    
    def remember_code(
        self,
        code: str,
        task: Optional[str] = None,
        file_path: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
        importance: float = 0.6
    ) -> str:
        """
        Store code related to a task.
        
        Args:
            code: The code content
            task: Related task name
            file_path: File path
            language: Programming language
            description: Description of what the code does
            importance: Importance score
            
        Returns:
            Node ID
        """
        tags = ["code", "task_related"]
        if task:
            tags.append(f"task:{task}")
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
                "task": task,
                "file_path": file_path,
                "language": language,
                "description": description
            }
        )
    
    def remember_decision(
        self,
        decision: str,
        task: Optional[str] = None,
        rationale: Optional[str] = None,
        importance: float = 0.8
    ) -> str:
        """
        Store a decision related to a task.
        
        Args:
            decision: The decision made
            task: Related task name
            rationale: Why this decision was made
            importance: Importance score
            
        Returns:
            Node ID
        """
        tags = ["decision", "task_related"]
        if task:
            tags.append(f"task:{task}")
        
        content = f"Decision: {decision}"
        if rationale:
            content += f"\nRationale: {rationale}"
        
        return self._adapter.remember(
            content=content,
            node_type="decision",
            tags=tags,
            importance=importance
        )
    
    def remember_error(
        self,
        error: str,
        solution: str,
        task: Optional[str] = None,
        importance: float = 0.9
    ) -> str:
        """
        Store an error and solution related to a task.
        
        Args:
            error: The error encountered
            solution: How it was solved
            task: Related task name
            importance: Importance score
            
        Returns:
            Node ID
        """
        tags = ["error", "solution", "task_related"]
        if task:
            tags.append(f"task:{task}")
        
        content = f"Error: {error}\nSolution: {solution}"
        
        return self._adapter.remember(
            content=content,
            node_type="solution",
            tags=tags,
            importance=importance
        )
    
    def recall_tasks(
        self,
        query: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for relevant tasks.
        
        Args:
            query: Search query
            status: Filter by status
            priority: Filter by priority
            limit: Max results
            
        Returns:
            List of task results
        """
        search_query = f"task {query}"
        if status:
            search_query += f" status:{status}"
        if priority:
            search_query += f" priority:{priority}"
        
        return self._adapter.recall(query=search_query, limit=limit)
    
    def recall_code(
        self,
        query: str,
        task: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """Search for relevant code."""
        search_query = f"code {query}"
        if task:
            search_query += f" task:{task}"
        
        return self._adapter.recall(query=search_query, limit=limit)
    
    def recall_decisions(
        self,
        query: str,
        task: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """Search for relevant decisions."""
        search_query = f"decision {query}"
        if task:
            search_query += f" task:{task}"
        
        return self._adapter.recall(query=search_query, limit=limit)
    
    def get_task_context(
        self,
        task: str,
        max_tokens: int = 2000
    ) -> str:
        """Get context for a specific task."""
        result = self._adapter.get_context(
            query=f"task:{task}",
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