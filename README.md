# Agent Memory System

A sophisticated multi-level hierarchical memory system for long-running AI agents that prevents context rot and maintains knowledge over weeks or months. **Now with adapters for 13+ agentic AI frameworks.**

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Adapter System](#adapter-system)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Integration](#framework-integration)
- [API Reference](#api-reference)
- [Node Types](#node-types)
- [Relation Types](#relation-types)
- [Memory Levels](#memory-levels)
- [Lossless Compaction](#lossless-compaction)
- [Context Assembly](#context-assembly)
- [Advanced Usage](#advanced-usage)
- [Performance](#performance)
- [File Structure](#file-structure)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

The Agent Memory System is designed to solve the **context rot problem** in long-running AI agents. Traditional approaches either:
1. Keep everything in context (limited by token window)
2. Summarize and lose information (lossy compression)
3. Simply discard old knowledge (information loss)

This system provides a **lossless** alternative using a multi-level hierarchy that automatically manages memory based on importance and access patterns.

### What's New: Framework Adapters

The system now includes **drop-in adapters** for the top agentic AI frameworks:

| Framework | Adapter | Integration Pattern |
|-----------|---------|---------------------|
| LangChain | `LangChainMemory` | BaseMemory interface |
| LangGraph | `LangGraphCheckpointer` | Checkpointer interface |
| OpenAI Agents | `MemoryTool` + `OpenAIAgentsMemory` | Tool + RunHooks |
| CrewAI | `CrewAIMemory` | Storage interface |
| AutoGen | `AutoGenMemory` | Memory interface |
| Semantic Kernel | `SemanticKernelMemory` | MemoryStoreBase |
| LlamaIndex | `LlamaIndexMemory` | BaseMemory interface |
| Agno | `AgnoMemory` | MemoryDb interface |
| Pydantic AI | `PydanticAIMemory` | Memory interface |
| Haystack | `HaystackMemory` | Component interface |
| OpenCode | `OpenCodeMemory` | Code agent memory |
| Cline | `ClineMemory` | Code agent memory |
| Task | `TaskMemory` | Task agent memory |

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FRAMEWORK ADAPTERS                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │LangChain │  │OpenAI    │  │CrewAI    │  │AutoGen   │  │   ...    │  │
│  │Memory    │  │Agents    │  │Memory    │  │Memory    │  │   ...    │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │             │             │             │             │         │
│       └─────────────┴─────────────┴─────────────┴─────────────┘         │
│                                    │                                     │
│                            ┌───────▼───────┐                             │
│                            │ BaseMemory    │                             │
│                            │ Adapter       │                             │
│                            └───────┬───────┘                             │
└────────────────────────────────────┼─────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼─────────────────────────────────────┐
│                        MEMORY MANAGER                                    │
│                            ┌───────▼───────┐                             │
│                            │ remember()    │                             │
│                            │ recall()      │                             │
│                            │ get_context() │                             │
│                            └───────┬───────┘                             │
│                                    │                                     │
└────────────────────────────────────┼─────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼─────────────────────────────────────┐
│                        STORAGE LAYERS                                    │
│       ┌────────────────────────────┼────────────────────────────┐        │
│       │                            │                            │        │
│  ┌────▼─────┐              ┌───────▼───────┐             ┌─────▼─────┐  │
│  │ LEVEL 1  │              │    LEVEL 2    │             │  LEVEL 3  │  │
│  │   RAM    │   compaction │     Warm      │  compaction │   Cold    │  │
│  │  (Hot)   │ ──────────► │   Storage     │ ──────────► │  Storage  │  │
│  │ 50-100   │              │  MD Files     │             │  MD Files │  │
│  │  nodes   │              │   Indexed     │             │ Archived  │  │
│  └──────────┘              └───────────────┘             └───────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
agent_memory/
├── __init__.py                    # Package exports
├── memory_manager.py              # Main API interface
├── knowledge_graph.py             # Core data structures
├── level_config.py                # Memory level presets
│
├── adapters/                      # Framework adapters
│   ├── __init__.py                # Lazy imports
│   ├── base.py                    # BaseMemoryAdapter
│   ├── langchain_adapter.py       # LangChain/LangGraph
│   ├── openai_agents_adapter.py   # OpenAI Agents SDK
│   ├── crewai_adapter.py          # CrewAI
│   ├── autogen_adapter.py         # AutoGen
│   ├── semantic_kernel_adapter.py # Semantic Kernel
│   ├── llamaindex_adapter.py      # LlamaIndex
│   ├── agno_adapter.py            # Agno
│   ├── pydantic_ai_adapter.py     # Pydantic AI
│   ├── haystack_adapter.py        # Haystack
│   ├── opencode_adapter.py        # OpenCode
│   ├── cline_adapter.py           # Cline
│   └── task_adapter.py            # Task
│
├── adk/                           # Google ADK integration
│   ├── __init__.py
│   ├── memory_tool.py
│   ├── memory_agent.py
│   └── memory_callback.py
│
├── storage/                       # Storage layers
│   ├── __init__.py
│   ├── ram_store.py               # L1 in-memory (LRU)
│   ├── file_store.py              # L2/L3 markdown files
│   └── index_manager.py           # Cross-level indexing
│
├── compaction/                    # Memory management
│   ├── __init__.py
│   ├── compactor.py               # Level migration logic
│   └── importance.py              # Scoring algorithms
│
└── retrieval/                     # Search & assembly
    ├── __init__.py
    ├── searcher.py                # Multi-level search
    └── assembler.py               # Context formatting
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER / FRAMEWORK                              │
│                                                                         │
│    agent.run("What's the API auth method?")                             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FRAMEWORK ADAPTER                                │
│                                                                         │
│  before_turn():                    after_turn():                        │
│  ┌─────────────────────┐           ┌─────────────────────┐              │
│  │ Get context for     │           │ Store interaction   │              │
│  │ current query       │           │ in memory           │              │
│  └──────────┬──────────┘           └──────────┬──────────┘              │
│             │                                 │                         │
└─────────────┼─────────────────────────────────┼─────────────────────────┘
              │                                 │
              ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        MEMORY MANAGER                                   │
│                                                                         │
│  get_context(query)              remember(content, type, tags)          │
│  ┌─────────────────────┐         ┌─────────────────────┐                │
│  │ 1. Search all levels│         │ 1. Create node      │                │
│  │ 2. Rank by relevance│         │ 2. Store in L1      │                │
│  │ 3. Assemble context │         │ 3. Update index     │                │
│  │ 4. Format for LLM   │         │ 4. Check compaction │                │
│  └─────────────────────┘         └─────────────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
              │                                 │
              ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        KNOWLEDGE GRAPH                                  │
│                                                                         │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐       │
│  │  Node    │────▶│  Edge    │────▶│  Node    │────▶│  Node    │       │
│  │  (fact)  │     │(causes)  │     │(error)   │     │(solution)│       │
│  └──────────┘     └──────────┘     └──────────┘     └──────────┘       │
│                                                                         │
│  Typed nodes with semantic relationships                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Lossless Compaction Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMPACTION CYCLE                                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: Importance Scoring                                      │    │
│  │                                                                 │    │
│  │ importance = recency_weight × recency_score                     │    │
│  │            + frequency_weight × frequency_score                 │    │
│  │            + connectivity_weight × connectivity_score           │    │
│  │            + type_weight × type_score                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: Compaction Triggers                                     │    │
│  │                                                                 │    │
│  │ IF L1.utilization > 80%:                                        │    │
│  │   compact_l1_to_l2(lowest_importance_nodes)                     │    │
│  │                                                                 │    │
│  │ IF L2.importance < threshold:                                   │    │
│  │   compact_l2_to_l3(least_important_nodes)                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: Lossless Migration                                      │    │
│  │                                                                 │    │
│  │ 1. Write FULL content to markdown file                          │    │
│  │ 2. Create index entry (file_path, byte_offset)                  │    │
│  │ 3. Remove from current level (KEEP index)                       │    │
│  │ 4. Update graph references                                      │    │
│  │                                                                 │    │
│  │ ✗ NO data is ever deleted                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4: Retrieval                                               │    │
│  │                                                                 │    │
│  │ 1. Search summaries in L1 index (always available)              │    │
│  │ 2. Load full content from file if needed                        │    │
│  │ 3. Promote accessed nodes back to L1                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Lossless Compaction
- **Never delete information** - only reorganize and index
- Full content preserved in markdown files
- Index entries maintain retrieval paths
- Human-readable storage format

### 2. Knowledge Graph
- Typed nodes (fact, decision, experience, error, etc.)
- Semantic relationships (causes, depends_on, supports, etc.)
- Graph traversal for related knowledge
- Automatic relationship discovery via shared tags

### 3. Importance Scoring
- **Recency**: Recently accessed nodes are more important
- **Frequency**: Frequently accessed nodes stay in memory
- **Connectivity**: Well-connected nodes are more important
- **Type**: Decisions and errors weighted higher
- **Decay**: Importance decays over time unless reinforced

### 4. Fast Retrieval
- O(1) access for L1 nodes
- Indexed search across all levels
- Tag-based filtering
- Type-based filtering
- Full-text search in summaries

### 5. Context Assembly
- Automatic LLM context generation
- Token budget management
- Summary and full content modes
- Related node inclusion

### 6. Persistence
- State survives across sessions
- JSON index for fast loading
- Markdown files for human readability
- Graceful corruption recovery

### 7. Framework Adapters
- Drop-in integration with popular frameworks
- Consistent API across all frameworks
- Lazy imports (frameworks only required when used)
- Multiple integration patterns (tools, hooks, storage, plugins)

## Adapter System

### Overview

The adapter system provides a unified interface for integrating the memory system with various agentic AI frameworks. All adapters inherit from `BaseMemoryAdapter` which provides shared logic for storing, retrieving, and injecting memory.

### BaseMemoryAdapter

The base adapter provides:

- `before_turn(message)` - Get context before LLM call
- `after_turn(message, response)` - Store interaction after LLM call
- `on_tool_call(tool_name, input, output)` - Store tool usage
- `on_error(error, context)` - Store errors
- `remember(content, node_type, tags, importance)` - Direct storage
- `recall(query, limit)` - Direct search
- `get_context(query, max_tokens)` - Get formatted context

### Integration Patterns

#### 1. Memory Interface (LangChain, LlamaIndex, Agno)

```python
from agent_memory.adapters import LangChainMemory

memory = LangChainMemory(base_path="./my_memory")

# Drop-in replacement for LangChain memory
agent = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory
)
```

#### 2. Tool-Based (OpenAI Agents, Pydantic AI)

```python
from agent_memory.adapters import MemoryTool

memory_tool = MemoryTool(base_path="./my_memory")

# Add memory tools to agent
agent = Agent(
    name="Assistant",
    tools=memory_tool.get_tools()  # remember, recall, get_context
)
```

#### 3. Hooks/Callbacks (OpenAI Agents, AutoGen)

```python
from agent_memory.adapters import OpenAIAgentsMemory

hooks = OpenAIAgentsMemory(base_path="./my_memory")

# Automatic memory on every turn
result = await Runner.run(agent, "Hello", hooks=hooks)
```

#### 4. Storage Interface (CrewAI)

```python
from agent_memory.adapters import CrewAIMemory

memory = CrewAIMemory(base_path="./my_memory")

crew = Crew(
    agents=[agent],
    tasks=[task],
    memory=True,
    short_term_memory=memory.short_term,
    long_term_memory=memory.long_term
)
```

#### 5. Plugin/Component (Semantic Kernel, Haystack)

```python
from agent_memory.adapters import SemanticKernelMemory

memory = SemanticKernelMemory(base_path="./my_memory")

kernel = Kernel()
kernel.add_memory_store(memory.get_store())
```

#### 6. Code Agent Memory (OpenCode, Cline, Task)

```python
from agent_memory.adapters import ClineMemory

memory = ClineMemory(base_path="./my_memory")

# Specialized for code agents
memory.remember_code(
    code="def authenticate(user, password): ...",
    file_path="auth.py",
    language="python"
)

memory.remember_decision(
    decision="Use JWT for auth",
    rationale="Stateless and scalable"
)

memory.remember_error(
    error="ImportError: No module named 'jwt'",
    solution="pip install PyJWT"
)
```

### Configuration

All adapters accept the same configuration options:

```python
from agent_memory.adapters import BaseMemoryAdapter, AdapterConfig

config = AdapterConfig(
    # Storage patterns
    store_on_user=True,
    store_on_assistant=False,
    store_on_tool=True,
    store_on_error=True,
    
    # Keywords that trigger storage
    store_keywords=["remember", "important", "decided"],
    
    # Importance defaults
    default_importance=0.5,
    decision_importance=0.8,
    error_importance=0.9,
    
    # Context injection
    inject_context=True,
    max_context_tokens=1500,
    context_position="system"
)

adapter = BaseMemoryAdapter(
    base_path="./my_memory",
    config=config,
    memory_preset="assistant"  # or "chatbot", "enterprise", "regulatory", "research"
)
```

## Installation

### Requirements

- Python 3.7+
- No external dependencies for core system
- Framework-specific dependencies for adapters

### Install

```bash
# Clone or download the agent_memory directory
# No pip install needed for core system

# Install framework adapters as needed
pip install langchain langchain-core    # For LangChain adapter
pip install openai-agents              # For OpenAI Agents adapter
pip install crewai                     # For CrewAI adapter
pip install autogen-agentchat          # For AutoGen adapter
pip install semantic-kernel            # For Semantic Kernel adapter
pip install llama-index-core           # For LlamaIndex adapter
pip install agno                       # For Agno adapter
pip install pydantic-ai                # For Pydantic AI adapter
pip install haystack-ai                # For Haystack adapter
```

### Verify Installation

```python
from agent_memory.adapters import list_adapters

# List all adapters and their status
adapters = list_adapters()
for name, info in adapters.items():
    status = "✓" if info["available"] else "✗"
    print(f"{status} {name}: {info['class']}")
```

## Quick Start

### Basic Usage

```python
from agent_memory import MemoryManager

# Initialize
memory = MemoryManager(base_path="./my_agent_memory")

# Store knowledge
node_id = memory.remember(
    content="The API uses JWT tokens for authentication",
    node_type="fact",
    tags=["api", "auth"],
    importance=0.8
)

# Search knowledge
results = memory.recall("authentication")
for r in results:
    print(f"{r['summary']} (score: {r['relevance_score']:.2f})")

# Get LLM context
context = memory.get_context(
    "how does authentication work?",
    max_tokens=2000
)
print(context["context"])
```

### Using Adapters

```python
from agent_memory.adapters import get_adapter

# Get adapter for your framework
memory = get_adapter("langchain", base_path="./my_memory")

# Or use specific adapter
from agent_memory.adapters import LangChainMemory
memory = LangChainMemory(
    base_path="./my_memory",
    memory_preset="assistant"
)

# All adapters support the same interface
memory.remember("Important fact", importance=0.8)
results = memory.recall("fact")
context = memory.get_context("tell me about facts")
```

### Creating Relationships

```python
# Store a decision
decision_id = memory.remember(
    content="We chose FastAPI for the backend",
    node_type="decision",
    tags=["architecture", "backend"],
    importance=0.9
)

# Store related fact with relationship
memory.remember_with_relation(
    content="FastAPI requires Python 3.7+",
    related_to=decision_id,
    relation_type="supports",
    tags=["fastapi", "requirements"]
)

# Get related nodes
related = memory.get_related(decision_id)
```

## Framework Integration

### LangChain

```python
from agent_memory.adapters import LangChainMemory

memory = LangChainMemory(
    base_path="./my_memory",
    memory_preset="assistant"
)

# Use as BaseMemory
from langchain.agents import AgentExecutor
agent = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory
)

# Direct access
memory.remember("Important fact")
results = memory.recall("fact")
```

### LangGraph

```python
from agent_memory.adapters import LangGraphCheckpointer

checkpointer = LangGraphCheckpointer(base_path="./my_memory")

# Use as checkpointer
graph = workflow.compile(checkpointer=checkpointer)

# Run with thread_id
config = {"configurable": {"thread_id": "user_123"}}
result = graph.invoke({"input": "hello"}, config)
```

### OpenAI Agents SDK

```python
from agent_memory.adapters import MemoryTool, OpenAIAgentsMemory

# Option 1: Use as tools
memory_tool = MemoryTool(base_path="./my_memory")
agent = Agent(
    name="Assistant",
    tools=memory_tool.get_tools()
)

# Option 2: Use as hooks for automatic memory
hooks = OpenAIAgentsMemory(base_path="./my_memory")
result = await Runner.run(agent, "Hello", hooks=hooks)
```

### CrewAI

```python
from agent_memory.adapters import CrewAIMemory

memory = CrewAIMemory(base_path="./my_memory")

crew = Crew(
    agents=[agent],
    tasks=[task],
    memory=True,
    short_term_memory=memory.short_term,
    long_term_memory=memory.long_term,
    entity_memory=memory.entity
)
```

### AutoGen

```python
from agent_memory.adapters import AutoGenMemory

memory = AutoGenMemory(base_path="./my_memory")

agent = AssistantAgent(
    name="Assistant",
    llm_config=llm_config,
    memory=[memory]
)
```

### Semantic Kernel

```python
from agent_memory.adapters import SemanticKernelMemory

memory = SemanticKernelMemory(base_path="./my_memory")

kernel = Kernel()
kernel.add_memory_store(memory.get_store())
```

### LlamaIndex

```python
from agent_memory.adapters import LlamaIndexMemory

memory = LlamaIndexMemory(base_path="./my_memory")

agent = OpenAIAgent(
    tools=tools,
    memory=memory
)
```

### Pydantic AI

```python
from agent_memory.adapters import PydanticAIMemory

memory = PydanticAIMemory(base_path="./my_memory")

agent = Agent(
    model="openai:gpt-4",
    memory=memory
)
```

### Code Agents (OpenCode, Cline, Task)

```python
from agent_memory.adapters import ClineMemory

memory = ClineMemory(base_path="./cline_memory")

# Store code context
memory.remember_code(
    code="def authenticate(user, password): ...",
    file_path="auth.py",
    language="python",
    description="JWT authentication function"
)

# Store decisions
memory.remember_decision(
    decision="Use PyJWT for token generation",
    rationale="Well-maintained, secure, simple API"
)

# Store errors and solutions
memory.remember_error(
    error="jwt.ExpiredSignatureError: Signature has expired",
    solution="Add token refresh logic and increase expiry time"
)

# Retrieve relevant code
results = memory.recall_code("authentication", limit=5)
```

## API Reference

### MemoryManager

#### `__init__(base_path, memory_preset, config, l1_capacity, l2_shard_size, l3_shard_size, auto_compact, compaction_threshold)`

Initialize the memory system.

**Parameters:**
- `base_path` (str): Directory for persistent storage (default: "./agent_memory_data")
- `memory_preset` (str): Preset name - "chatbot", "assistant", "enterprise", "regulatory", "research"
- `config` (MemoryConfig): Custom configuration (overrides preset)
- `l1_capacity` (int): [Legacy] Maximum nodes in RAM (default: 100)
- `l2_shard_size` (int): [Legacy] Nodes per L2 file (default: 50)
- `l3_shard_size` (int): [Legacy] Nodes per L3 file (default: 100)
- `auto_compact` (bool): Enable automatic compaction (default: True)
- `compaction_threshold` (float): L1 utilization to trigger compaction (default: 0.8)

#### `remember(content, node_type, tags, summary, importance, metadata)`

Store a new piece of knowledge.

**Parameters:**
- `content` (str): The knowledge content (required)
- `node_type` (str): Type of knowledge (default: "fact")
- `tags` (List[str]): Tags for categorization (default: [])
- `summary` (str): Optional summary (auto-generated if not provided)
- `importance` (float): Initial importance 0-1 (default: 0.5)
- `metadata` (Dict): Additional metadata (default: {})

**Returns:** Node ID (str)

#### `recall(query, limit, load_full)`

Search for knowledge matching a query.

**Parameters:**
- `query` (str): Search query
- `limit` (int): Maximum results (default: 10)
- `load_full` (bool): Load full content (default: False)

**Returns:** List of result dictionaries with node_id, summary, relevance_score, storage_level, content, was_promoted

#### `get(node_id)` / `get_context(query, max_tokens)` / `get_working_memory()` / `search_by_tag(tag, limit)` / `search_by_type(node_type, limit)` / `get_related(node_id, limit)` / `get_recent(limit)` / `add_relation(source_id, target_id, relation_type, weight)` / `update_importance(node_id, importance)` / `force_compact()` / `save()` / `get_stats()` / `get_memory_summary()` / `clear()` / `export_markdown(output_path)`

See detailed API documentation above.

### BaseMemoryAdapter

All framework adapters inherit from this base class and provide:

- `before_turn(message: str) -> Optional[str]` - Get context before LLM call
- `after_turn(message: str, response: str, metadata: Optional[Dict] = None) -> None` - Store interaction
- `on_tool_call(tool_name: str, tool_input: Dict, tool_output: Any) -> None` - Store tool usage
- `on_error(error: Exception, context: str = "") -> None` - Store error
- `remember(content: str, **kwargs) -> str` - Store knowledge
- `recall(query: str, limit: int = 5) -> List[Dict]` - Search memory
- `get_context(query: str, max_tokens: int = 2000) -> Dict` - Get formatted context
- `save() -> None` - Save to disk
- `get_stats() -> Dict` - Get statistics

## Memory Levels

### Presets

| Preset | Levels | L1 Capacity | Use Case |
|--------|--------|-------------|----------|
| `chatbot` | 2 | 50 | Simple chatbot |
| `assistant` | 3 | 100 | General assistant (default) |
| `enterprise` | 5 | 200 | Large-scale enterprise |
| `regulatory` | 7 | 150 | Compliance/audit |
| `research` | 4 | 100 | Research agent |

### Level Details

| Level | Storage | Capacity | Access | Content |
|-------|---------|----------|--------|---------|
| L1 (Hot) | RAM | 50-200 | O(1) | Full content |
| L2 (Warm) | Files | Unlimited | O(1) index | Full content |
| L3 (Cold) | Files | Unlimited | O(1) index | Full content |

## Lossless Compaction

### Guarantee

**No information is ever deleted.** Every piece of knowledge is preserved in markdown files with exact file locations stored in the index. The system only reorganizes - never destroys.

### Process

1. **Importance Scoring**: Each node is scored based on recency, frequency, connectivity, and type
2. **Compaction Triggers**: L1 > 80% capacity → Compact L1 to L2; L2 importance < threshold → Compact L2 to L3
3. **Lossless Migration**: Write FULL content to file, create index entry, remove from current level (keep index)
4. **Retrieval**: Search summaries in L1 index, load full content from file if needed, promote accessed nodes

## Context Assembly

### For LLM Prompts

```python
context = memory.get_context(
    "how does authentication work?",
    max_tokens=2000,
    include_summaries=True,
    include_full_content=True
)

system_prompt = f"""
You are an AI assistant with access to the following knowledge:

{context["context"]}

Based on this knowledge, answer the user's question.
"""
```

## Best Practices

### 1. Use Meaningful Tags
### 2. Write Good Summaries
### 3. Set Appropriate Importance
### 4. Create Relationships
### 5. Run Periodic Maintenance
### 6. Use Memory Presets
### 7. Leverage Framework Adapters

## Troubleshooting

### Common Issues

1. **Nodes Not Found After Compaction**: System now properly stores evicted nodes to L2
2. **Slow Performance**: Check L1 capacity, force compaction, use SSD
3. **File Corruption**: System handles gracefully with index rebuilding
4. **Memory Usage Too High**: Reduce L1 capacity, force compaction
5. **Search Returns No Results**: Check stats, try broader terms, search by tag

## License

MIT License

---

**Built with ❤️ for the agentic AI community**