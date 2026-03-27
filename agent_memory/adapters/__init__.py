"""
Memory Adapters for Agentic AI Frameworks.

Provides lossless memory adapters for the top agentic AI frameworks:
- LangChain / LangGraph
- OpenAI Agents SDK
- CrewAI
- AutoGen
- Semantic Kernel
- LlamaIndex
- Agno
- Pydantic AI
- Haystack
- OpenCode
- Cline
- Task

All adapters inherit from BaseMemoryAdapter for consistent behavior.
"""

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Always available
from .base import BaseMemoryAdapter, AdapterConfig, StorePattern

# Lazy imports for framework-specific adapters
__all__ = [
    # Base
    "BaseMemoryAdapter",
    "AdapterConfig",
    "StorePattern",
    # LangChain
    "LangChainMemory",
    "LangGraphCheckpointer",
    # OpenAI Agents
    "OpenAIAgentsMemory",
    "MemoryTool",
    # CrewAI
    "CrewAIMemory",
    "CrewAIStorage",
    # AutoGen
    "AutoGenMemory",
    "AutoGenConversableMemory",
    # Semantic Kernel
    "SemanticKernelMemory",
    "SemanticKernelMemoryStore",
    # LlamaIndex
    "LlamaIndexMemory",
    "LlamaIndexChatMemoryBuffer",
    # Agno
    "AgnoMemory",
    "AgnoMemoryDb",
    "AgnoAssistantMemory",
    # Pydantic AI
    "PydanticAIMemory",
    "PydanticAIToolMemory",
    "PydanticAIDepsMemory",
    # Haystack
    "HaystackMemory",
    "HaystackMemoryComponent",
    "HaystackChatMemory",
    # Code Agents
    "OpenCodeMemory",
    "ClineMemory",
    "TaskMemory",
]


def __getattr__(name: str):
    """Lazy import adapters to avoid import errors when frameworks aren't installed."""
    
    # LangChain
    if name in ("LangChainMemory", "LangGraphCheckpointer"):
        try:
            from .langchain_adapter import LangChainMemory, LangGraphCheckpointer
            return LangChainMemory if name == "LangChainMemory" else LangGraphCheckpointer
        except ImportError:
            raise ImportError(
                f"Cannot import {name}. LangChain is required. "
                "Install with: pip install langchain langchain-core"
            )
    
    # OpenAI Agents
    if name in ("OpenAIAgentsMemory", "MemoryTool"):
        try:
            from .openai_agents_adapter import OpenAIAgentsMemory, MemoryTool
            return OpenAIAgentsMemory if name == "OpenAIAgentsMemory" else MemoryTool
        except ImportError:
            raise ImportError(
                f"Cannot import {name}. OpenAI Agents SDK is required. "
                "Install with: pip install openai-agents"
            )
    
    # CrewAI
    if name in ("CrewAIMemory", "CrewAIStorage"):
        try:
            from .crewai_adapter import CrewAIMemory, CrewAIStorage
            return CrewAIMemory if name == "CrewAIMemory" else CrewAIStorage
        except ImportError:
            raise ImportError(
                f"Cannot import {name}. CrewAI is required. "
                "Install with: pip install crewai"
            )
    
    # AutoGen
    if name in ("AutoGenMemory", "AutoGenConversableMemory"):
        try:
            from .autogen_adapter import AutoGenMemory, AutoGenConversableMemory
            return AutoGenMemory if name == "AutoGenMemory" else AutoGenConversableMemory
        except ImportError:
            raise ImportError(
                f"Cannot import {name}. AutoGen is required. "
                "Install with: pip install autogen-agentchat"
            )
    
    # Semantic Kernel
    if name in ("SemanticKernelMemory", "SemanticKernelMemoryStore"):
        try:
            from .semantic_kernel_adapter import SemanticKernelMemory, SemanticKernelMemoryStore
            return SemanticKernelMemory if name == "SemanticKernelMemory" else SemanticKernelMemoryStore
        except ImportError:
            raise ImportError(
                f"Cannot import {name}. Semantic Kernel is required. "
                "Install with: pip install semantic-kernel"
            )
    
    # LlamaIndex
    if name in ("LlamaIndexMemory", "LlamaIndexChatMemoryBuffer"):
        try:
            from .llamaindex_adapter import LlamaIndexMemory, LlamaIndexChatMemoryBuffer
            return LlamaIndexMemory if name == "LlamaIndexMemory" else LlamaIndexChatMemoryBuffer
        except ImportError:
            raise ImportError(
                f"Cannot import {name}. LlamaIndex is required. "
                "Install with: pip install llama-index-core"
            )
    
    # Agno
    if name in ("AgnoMemory", "AgnoMemoryDb", "AgnoAssistantMemory"):
        try:
            from .agno_adapter import AgnoMemory, AgnoMemoryDb, AgnoAssistantMemory
            if name == "AgnoMemory":
                return AgnoMemory
            elif name == "AgnoMemoryDb":
                return AgnoMemoryDb
            else:
                return AgnoAssistantMemory
        except ImportError:
            raise ImportError(
                f"Cannot import {name}. Agno is required. "
                "Install with: pip install agno"
            )
    
    # Pydantic AI
    if name in ("PydanticAIMemory", "PydanticAIToolMemory", "PydanticAIDepsMemory"):
        try:
            from .pydantic_ai_adapter import PydanticAIMemory, PydanticAIToolMemory, PydanticAIDepsMemory
            if name == "PydanticAIMemory":
                return PydanticAIMemory
            elif name == "PydanticAIToolMemory":
                return PydanticAIToolMemory
            else:
                return PydanticAIDepsMemory
        except ImportError:
            raise ImportError(
                f"Cannot import {name}. Pydantic AI is required. "
                "Install with: pip install pydantic-ai"
            )
    
    # Haystack
    if name in ("HaystackMemory", "HaystackMemoryComponent", "HaystackChatMemory"):
        try:
            from .haystack_adapter import HaystackMemory, HaystackMemoryComponent, HaystackChatMemory
            if name == "HaystackMemory":
                return HaystackMemory
            elif name == "HaystackMemoryComponent":
                return HaystackMemoryComponent
            else:
                return HaystackChatMemory
        except ImportError:
            raise ImportError(
                f"Cannot import {name}. Haystack is required. "
                "Install with: pip install haystack-ai"
            )
    
    # Code Agents (no external dependencies)
    if name == "OpenCodeMemory":
        from .opencode_adapter import OpenCodeMemory
        return OpenCodeMemory
    
    if name == "ClineMemory":
        from .cline_adapter import ClineMemory
        return ClineMemory
    
    if name == "TaskMemory":
        from .task_adapter import TaskMemory
        return TaskMemory
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_adapter(framework: str, **kwargs) -> BaseMemoryAdapter:
    """
    Get an adapter for a specific framework.
    
    Args:
        framework: Framework name (langchain, openai, crewai, autogen, etc.)
        **kwargs: Adapter configuration
        
    Returns:
        Configured adapter instance
        
    Example:
        from agent_memory.adapters import get_adapter
        
        # Get a LangChain adapter
        memory = get_adapter("langchain", base_path="./my_memory")
        
        # Get an OpenAI Agents adapter
        memory = get_adapter("openai", base_path="./my_memory")
    """
    framework = framework.lower().replace("-", "_").replace(" ", "_")
    
    adapters = {
        "langchain": "LangChainMemory",
        "langgraph": "LangGraphCheckpointer",
        "openai": "OpenAIAgentsMemory",
        "openai_agents": "OpenAIAgentsMemory",
        "crewai": "CrewAIMemory",
        "autogen": "AutoGenMemory",
        "semantic_kernel": "SemanticKernelMemory",
        "semantickernel": "SemanticKernelMemory",
        "llamaindex": "LlamaIndexMemory",
        "llama_index": "LlamaIndexMemory",
        "agno": "AgnoMemory",
        "pydantic_ai": "PydanticAIMemory",
        "pydanticai": "PydanticAIMemory",
        "haystack": "HaystackMemory",
        "opencode": "OpenCodeMemory",
        "cline": "ClineMemory",
        "task": "TaskMemory",
    }
    
    if framework not in adapters:
        raise ValueError(
            f"Unknown framework: {framework}. "
            f"Supported frameworks: {', '.join(adapters.keys())}"
        )
    
    adapter_class_name = adapters[framework]
    
    # Use __getattr__ to trigger lazy import
    adapter_class = globals().get(adapter_class_name)
    if adapter_class is None:
        # Trigger lazy import
        adapter_class = __getattr__(adapter_class_name)
    
    return adapter_class(**kwargs)


def list_adapters() -> dict:
    """
    List all available adapters and their status.
    
    Returns:
        Dict mapping adapter names to availability status
    """
    results = {
        "base": {"available": True, "class": "BaseMemoryAdapter"},
    }
    
    # Check each framework
    checks = {
        "langchain": ("LangChainMemory", "langchain"),
        "langgraph": ("LangGraphCheckpointer", "langchain"),
        "openai_agents": ("OpenAIAgentsMemory", "openai-agents"),
        "crewai": ("CrewAIMemory", "crewai"),
        "autogen": ("AutoGenMemory", "autogen-agentchat"),
        "semantic_kernel": ("SemanticKernelMemory", "semantic-kernel"),
        "llamaindex": ("LlamaIndexMemory", "llama-index-core"),
        "agno": ("AgnoMemory", "agno"),
        "pydantic_ai": ("PydanticAIMemory", "pydantic-ai"),
        "haystack": ("HaystackMemory", "haystack-ai"),
        "opencode": ("OpenCodeMemory", None),
        "cline": ("ClineMemory", None),
        "task": ("TaskMemory", None),
    }
    
    for name, (class_name, package) in checks.items():
        try:
            if package:
                __import__(package)
            results[name] = {
                "available": True,
                "class": class_name,
                "package": package
            }
        except ImportError:
            results[name] = {
                "available": False,
                "class": class_name,
                "package": package,
                "install": f"pip install {package}" if package else None
            }
    
    return results