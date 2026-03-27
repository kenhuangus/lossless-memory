"""
Agent Memory System - Multi-level hierarchical memory for long-running agents.

This package implements a sophisticated memory system that allows AI agents
to maintain knowledge over weeks or months without context rot.

Key Features:
- Flexible N-level memory hierarchy (2-7 levels)
- Lossless compaction - no information is ever deleted
- Knowledge graph with typed edges
- Importance-based memory management
- Fast retrieval across all levels
- Context assembly for LLM prompts
- Preset configurations for common use cases

Usage:
    from agent_memory import MemoryManager
    
    # Simple 3-level memory (default)
    memory = MemoryManager(base_path="./my_agent_memory")
    
    # Use a preset configuration
    memory = MemoryManager(
        base_path="./my_agent_memory",
        memory_preset="enterprise"  # 5 levels for long-running agents
    )
    
    # Custom configuration
    from agent_memory import create_custom_config
    config = create_custom_config(num_levels=4, hot_capacity=200)
    memory = MemoryManager(base_path="./my_agent_memory", config=config)
    
    # Store knowledge
    node_id = memory.remember(
        content="Important fact about the system",
        node_type="fact",
        tags=["system", "important"]
    )
    
    # Recall knowledge
    results = memory.recall("system facts")
    
    # Get context for LLM
    context = memory.get_context("what do we know about the system?")
"""

from .knowledge_graph import (
    KnowledgeNode,
    KnowledgeGraph,
    Edge,
    NodeType,
    RelationType,
    StorageLevel
)
from .memory_manager import MemoryManager
from .level_config import (
    MemoryConfig,
    LevelSpec,
    StorageType,
    get_preset,
    list_presets,
    create_custom_config,
    create_chatbot_config,
    create_assistant_config,
    create_enterprise_config,
    create_regulatory_config,
    create_research_config
)

__version__ = "1.1.0"

__all__ = [
    # Core classes
    'MemoryManager',
    'KnowledgeNode',
    'KnowledgeGraph',
    'Edge',
    'NodeType',
    'RelationType',
    'StorageLevel',
    
    # Configuration
    'MemoryConfig',
    'LevelSpec',
    'StorageType',
    'get_preset',
    'list_presets',
    'create_custom_config',
    
    # Preset factories
    'create_chatbot_config',
    'create_assistant_config',
    'create_enterprise_config',
    'create_regulatory_config',
    'create_research_config'
]
