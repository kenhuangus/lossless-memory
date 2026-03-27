
"""
LangChain / LangGraph Memory Adapter.

Provides lossless memory for LangChain agents and LangGraph workflows.

Usage:
    from agent_memory.adapters import LangChainMemory
    
    # Option 1: Use as LangChain BaseMemory
    memory = LangChainMemory(base_path="./my_memory")
    agent = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory  # Drop-in replacement
    )
    
    # Option 2: Use with LangGraph checkpointer
    from agent_memory.adapters import LangGraphCheckpointer
    checkpointer = LangGraphCheckpointer(base_path="./my_memory")
    graph = workflow.compile(checkpointer=checkpointer)
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.memory import BaseMemory
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseMemory = object
    BaseChatMemory = object
    BaseMessage = object

from .base import BaseMemoryAdapter, AdapterConfig
from ..memory_manager import MemoryManager
from ..knowledge_graph import NodeType

logger = logging.getLogger(__name__)


class LangChainMemory(BaseMemory):
    """
    LangChain BaseMemory implementation using lossless memory system.
    
    Drop-in replacement for LangChain's built-in memory classes.
    Provides lossless multi-level memory instead of simple buffer/window memory.
    
    Usage:
        from agent_memory.adapters import LangChainMemory
        
        memory = LangChainMemory(
            base_path="./my_memory",
            memory_preset="assistant"
        )
        
        # Use with any LangChain agent
        agent = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory
        )
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./langchain_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        **kwargs
    ):
        """Initialize LangChain memory adapter."""
        if not HAS_LANGCHAIN:
            raise ImportError(
                "LangChain is required for LangChainMemory. "
                "Install with: pip install langchain langchain-core"
            )
        
        super().__init__()
        
        # Initialize base adapter
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        # Store as instance attributes (using object.__setattr__ for Pydantic compatibility)
        object.__setattr__(self, 'memory_key', memory_key)
        object.__setattr__(self, 'input_key', input_key)
        object.__setattr__(self, 'output_key', output_key)
        object.__setattr__(self, 'return_messages', return_messages)
        
        # Store conversation history for LangChain compatibility
        object.__setattr__(self, 'chat_memory', [])
        
        logger.info("LangChainMemory initialized")
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for the chain."""
        query = inputs.get(self.input_key, "")
        
        # Get context from our memory system
        context = self._adapter.before_turn(query)
        
        if self.return_messages:
            # Return as messages
            if context:
                return {self.memory_key: [SystemMessage(content=context)]}
            return {self.memory_key: []}
        else:
            # Return as string
            if context:
                return {self.memory_key: context}
            return {self.memory_key: ""}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from the chain interaction."""
        input_text = inputs.get(self.input_key, "")
        output_text = outputs.get(self.output_key, "")
        
        # Store in our memory system
        self._adapter.after_turn(
            message=input_text,
            response=output_text
        )
        
        # Also keep in chat_memory for LangChain compatibility
        if HAS_LANGCHAIN:
            self.chat_memory.append(HumanMessage(content=input_text))
            self.chat_memory.append(AIMessage(content=output_text))
    
    def clear(self) -> None:
        """Clear memory."""
        self._adapter.clear()
        self.chat_memory = []
    
    # Direct access to our memory system
    def remember(self, content: str, **kwargs) -> str:
        """Store knowledge directly."""
        return self._adapter.remember(content, **kwargs)
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory directly."""
        return self._adapter.recall(query, limit)
    
    def get_context(self, query: str, max_tokens: int = 2000) -> Dict:
        """Get formatted context for LLM."""
        return self._adapter.get_context(query=query, max_tokens=max_tokens)
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self._adapter.get_stats()
    
    def save(self) -> None:
        """Save memory to disk."""
        self._adapter.save()


class LangGraphCheckpointer:
    """
    LangGraph checkpointer using lossless memory.
    
    Use with LangGraph's compiled graphs to persist state
    across conversations using our lossless memory system.
    
    Usage:
        from agent_memory.adapters import LangGraphCheckpointer
        
        checkpointer = LangGraphCheckpointer(base_path="./my_memory")
        graph = workflow.compile(checkpointer=checkpointer)
        
        # Run with thread_id
        config = {"configurable": {"thread_id": "user_123"}}
        result = graph.invoke({"input": "hello"}, config)
    """
    
    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        base_path: str = "./langgraph_memory",
        memory_preset: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
        **kwargs
    ):
        """Initialize LangGraph checkpointer."""
        self._adapter = BaseMemoryAdapter(
            memory=memory,
            config=config,
            base_path=base_path,
            memory_preset=memory_preset
        )
        
        # In-memory state cache for current session
        self._state_cache: Dict[str, Dict] = {}
        
        logger.info("LangGraphCheckpointer initialized")
    
    def get(self, config: Dict) -> Optional[Dict]:
        """Get checkpoint for a thread."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        
        # Check cache first
        if thread_id in self._state_cache:
            return self._state_cache[thread_id]
        
        # Try to load from memory
        nodes = self._adapter.recall(f"thread:{thread_id}", limit=10)
        if nodes:
            # Reconstruct state from stored nodes
            state = {}
            for node in nodes:
                if node.get("content", "").startswith("state:"):
                    import json
                    try:
                        state_str = node["content"][6:]  # Remove "state:" prefix
                        state = json.loads(state_str)
                    except (json.JSONDecodeError, ValueError):
                        pass
            
            if state:
                self._state_cache[thread_id] = state
                return state
        
        return None
    
    def put(self, config: Dict, checkpoint: Dict) -> None:
        """Save checkpoint for a thread."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        
        # Cache in memory
        self._state_cache[thread_id] = checkpoint
        
        # Store in memory system
        import json
        state_str = json.dumps(checkpoint, default=str)
        
        self._adapter.remember(
            content=f"state:{state_str}",
            node_type="experience",
            tags=["checkpoint", f"thread:{thread_id}"],
            importance=0.3
        )
    
    def list(self, config: Dict, **kwargs) -> List[Dict]:
        """List checkpoints for a thread."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        
        nodes = self._adapter.recall(f"thread:{thread_id}", limit=50)
        
        checkpoints = []
        for node in nodes:
            if node.get("content", "").startswith("state:"):
                checkpoints.append({
                    "thread_id": thread_id,
                    "checkpoint_id": node.get("node_id", ""),
                    "created": node.get("summary", "")
                })
        
        return checkpoints
    
    def put_writes(self, config: Dict, writes: List, task_id: str) -> None:
        """Save intermediate writes."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        
        import json
        writes_str = json.dumps(writes, default=str)
        
        self._adapter.remember(
            content=f"writes:{writes_str}",
            node_type="observation",
            tags=["writes", f"thread:{thread_id}", f"task:{task_id}"],
            importance=0.2
        )
    
    # Direct access
    def remember(self, content: str, **kwargs) -> str:
        """Store knowledge directly."""
        return self._adapter.remember(content, **kwargs)
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory directly."""
        return self._adapter.recall(query, limit)
    
    def save(self) -> None:
        """Save memory to disk."""
        self._adapter.save()
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self._adapter.get_stats()