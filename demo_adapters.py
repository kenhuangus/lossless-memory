"""
Demo script showing usage patterns for all memory adapters.

Demonstrates how to use each adapter with its framework.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_memory.adapters import (
    BaseMemoryAdapter,
    get_adapter,
    list_adapters,
)


def demo_base():
    """Demo: Base memory adapter."""
    print("\n" + "="*60)
    print("DEMO: BaseMemoryAdapter")
    print("="*60)
    
    memory = BaseMemoryAdapter(base_path="./demo_memory/base")
    
    # Store knowledge
    memory.remember(
        content="The API uses JWT tokens for authentication",
        node_type="fact",
        tags=["api", "auth"],
        importance=0.8
    )
    
    # Search memory
    results = memory.recall("authentication", limit=5)
    print(f"Found {len(results)} results")
    
    # Get context for LLM
    context = memory.get_context("how to authenticate", max_tokens=1000)
    print(f"Context: {context['context'][:100]}...")
    
    # Clean up
    import shutil
    shutil.rmtree("./demo_memory/base", ignore_errors=True)


def demo_code_agents():
    """Demo: Code agent adapters (OpenCode, Cline, Task)."""
    print("\n" + "="*60)
    print("DEMO: Code Agent Adapters")
    print("="*60)
    
    from agent_memory.adapters import OpenCodeMemory, ClineMemory, TaskMemory
    
    for name, AdapterClass in [
        ("OpenCode", OpenCodeMemory),
        ("Cline", ClineMemory),
        ("Task", TaskMemory),
    ]:
        print(f"\n--- {name} ---")
        memory = AdapterClass(base_path=f"./demo_memory/{name.lower()}")
        
        # Store code
        memory.remember_code(
            code="def authenticate(user, password):\n    return jwt.encode({'user': user}, SECRET)",
            file_path="auth.py",
            language="python",
            description="JWT authentication function"
        )
        
        # Store decision
        memory.remember_decision(
            decision="Use PyJWT for token generation",
            rationale="Well-maintained, secure, simple API"
        )
        
        # Store error/solution
        memory.remember_error(
            error="jwt.ExpiredSignatureError: Signature has expired",
            solution="Add token refresh logic and increase expiry time"
        )
        
        # Search
        results = memory.recall_code("authentication", limit=3)
        print(f"  Found {len(results)} code results")
        
        # Clean up
        import shutil
        shutil.rmtree(f"./demo_memory/{name.lower()}", ignore_errors=True)


def demo_langchain():
    """Demo: LangChain memory adapter."""
    print("\n" + "="*60)
    print("DEMO: LangChain Memory")
    print("="*60)
    
    try:
        from agent_memory.adapters import LangChainMemory
        
        memory = LangChainMemory(
            base_path="./demo_memory/langchain",
            memory_preset="assistant"
        )
        
        # Simulate chain interaction
        memory.save_context(
            inputs={"input": "What is Python?"},
            outputs={"output": "Python is a high-level programming language."}
        )
        
        # Load memory variables
        result = memory.load_memory_variables({"input": "Tell me about Python"})
        print(f"Memory context: {str(result)[:100]}...")
        
        # Direct access
        memory.remember("Python was created by Guido van Rossum", importance=0.8)
        results = memory.recall("Python creator", limit=5)
        print(f"Found {len(results)} results")
        
        # Clean up
        import shutil
        shutil.rmtree("./demo_memory/langchain", ignore_errors=True)
        
    except ImportError:
        print("  ⚠️  LangChain not installed")


def demo_openai_agents():
    """Demo: OpenAI Agents SDK memory."""
    print("\n" + "="*60)
    print("DEMO: OpenAI Agents Memory")
    print("="*60)
    
    try:
        from agent_memory.adapters import MemoryTool, OpenAIAgentsMemory
        
        # Option 1: Use as tools
        memory_tool = MemoryTool(base_path="./demo_memory/openai")
        tools = memory_tool.get_tools()
        print(f"  Created {len(tools)} memory tools: {[t.name for t in tools]}")
        
        # Option 2: Use as hooks
        hooks = OpenAIAgentsMemory(base_path="./demo_memory/openai2")
        hooks.remember("Important fact for the agent", importance=0.9)
        results = hooks.recall("important", limit=5)
        print(f"  Hooks stored and retrieved {len(results)} items")
        
        # Clean up
        import shutil
        shutil.rmtree("./demo_memory/openai", ignore_errors=True)
        shutil.rmtree("./demo_memory/openai2", ignore_errors=True)
        
    except ImportError:
        print("  ⚠️  OpenAI Agents SDK not installed")


def demo_crewai():
    """Demo: CrewAI memory."""
    print("\n" + "="*60)
    print("DEMO: CrewAI Memory")
    print("="*60)
    
    try:
        from agent_memory.adapters import CrewAIMemory
        
        memory = CrewAIMemory(base_path="./demo_memory/crewai")
        
        # Use with crew's memory types
        memory.short_term.save("User prefers dark mode", metadata={"type": "preference"})
        memory.long_term.save("Project uses React", metadata={"type": "tech"})
        memory.entity.save("John is the project manager", metadata={"entity": "person"})
        
        # Direct access
        memory.remember("Important crew decision", importance=0.8)
        results = memory.recall("decision", limit=5)
        print(f"  Found {len(results)} results")
        
        # Clean up
        import shutil
        shutil.rmtree("./demo_memory/crewai", ignore_errors=True)
        
    except ImportError:
        print("  ⚠️  CrewAI not installed")


def demo_autogen():
    """Demo: AutoGen memory."""
    print("\n" + "="*60)
    print("DEMO: AutoGen Memory")
    print("="*60)
    
    try:
        from agent_memory.adapters import AutoGenMemory
        
        memory = AutoGenMemory(base_path="./demo_memory/autogen")
        
        # Use with AutoGen's memory interface
        import asyncio
        
        async def demo():
            from autogen_core.memory import MemoryContent, MemoryMimeType
            
            # Add memory
            content = MemoryContent(
                content="The project deadline is December 15",
                mime_type=MemoryMimeType.TEXT
            )
            await memory.add(content)
            
            # Query memory
            results = await memory.query("project deadline", top_k=5)
            print(f"  Found {len(results)} results")
        
        asyncio.run(demo())
        
        # Clean up
        import shutil
        shutil.rmtree("./demo_memory/autogen", ignore_errors=True)
        
    except ImportError:
        print("  ⚠️  AutoGen not installed")


def demo_semantic_kernel():
    """Demo: Semantic Kernel memory."""
    print("\n" + "="*60)
    print("DEMO: Semantic Kernel Memory")
    print("="*60)
    
    try:
        from agent_memory.adapters import SemanticKernelMemory
        
        memory = SemanticKernelMemory(base_path="./demo_memory/sk")
        
        # Use as memory store
        store = memory.get_store()
        store.remember("Azure uses managed identity for auth", importance=0.7)
        
        results = store.recall("Azure authentication", limit=5)
        print(f"  Found {len(results)} results")
        
        # Clean up
        import shutil
        shutil.rmtree("./demo_memory/sk", ignore_errors=True)
        
    except ImportError:
        print("  ⚠️  Semantic Kernel not installed")


def demo_llamaindex():
    """Demo: LlamaIndex memory."""
    print("\n" + "="*60)
    print("DEMO: LlamaIndex Memory")
    print("="*60)
    
    try:
        from agent_memory.adapters import LlamaIndexMemory
        
        memory = LlamaIndexMemory(base_path="./demo_memory/llama")
        
        # Use with chat history
        from llama_index.core.llms import ChatMessage, MessageRole
        
        memory.put(ChatMessage(role=MessageRole.USER, content="What is RAG?"))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="RAG combines retrieval with generation."))
        
        # Get messages with context
        messages = memory.get(input="Tell me about RAG")
        print(f"  Got {len(messages)} messages")
        
        # Clean up
        import shutil
        shutil.rmtree("./demo_memory/llama", ignore_errors=True)
        
    except ImportError:
        print("  ⚠️  LlamaIndex not installed")


def demo_agno():
    """Demo: Agno memory."""
    print("\n" + "="*60)
    print("DEMO: Agno Memory")
    print("="*60)
    
    try:
        from agent_memory.adapters import AgnoMemory
        
        memory = AgnoMemory(base_path="./demo_memory/agno")
        
        # Use with Agno's memory interface
        memory.db.remember("User prefers email notifications", importance=0.6)
        
        results = memory.db.recall("notifications", limit=5)
        print(f"  Found {len(results)} results")
        
        # Clean up
        import shutil
        shutil.rmtree("./demo_memory/agno", ignore_errors=True)
        
    except ImportError:
        print("  ⚠️  Agno not installed")


def demo_pydantic_ai():
    """Demo: Pydantic AI memory."""
    print("\n" + "="*60)
    print("DEMO: Pydantic AI Memory")
    print("="*60)
    
    try:
        from agent_memory.adapters import PydanticAIMemory, PydanticAIDepsMemory
        
        # Option 1: Direct memory
        memory = PydanticAIMemory(base_path="./demo_memory/pydantic")
        memory.add_message("user", "What is dependency injection?")
        memory.add_message("assistant", "DI is a design pattern where dependencies are provided externally.")
        
        context = memory.get_context_for_prompt("dependency injection")
        print(f"  Context: {context[:80]}...")
        
        # Option 2: As dependency
        deps = PydanticAIDepsMemory(base_path="./demo_memory/pydantic2")
        deps.remember("Important fact", importance=0.8)
        results = deps.recall("important", limit=5)
        print(f"  Deps found {len(results)} results")
        
        # Clean up
        import shutil
        shutil.rmtree("./demo_memory/pydantic", ignore_errors=True)
        shutil.rmtree("./demo_memory/pydantic2", ignore_errors=True)
        
    except ImportError:
        print("  ⚠️  Pydantic AI not installed")


def demo_haystack():
    """Demo: Haystack memory."""
    print("\n" + "="*60)
    print("DEMO: Haystack Memory")
    print("="*60)
    
    try:
        from agent_memory.adapters import HaystackMemory
        
        memory = HaystackMemory(base_path="./demo_memory/haystack")
        
        # Add messages
        memory.add({"role": "user", "content": "Explain transformers"})
        memory.add({"role": "assistant", "content": "Transformers use self-attention mechanisms."})
        
        # Search
        docs = memory.search("transformers", top_k=5)
        print(f"  Found {len(docs)} documents")
        
        # Clean up
        import shutil
        shutil.rmtree("./demo_memory/haystack", ignore_errors=True)
        
    except ImportError:
        print("  ⚠️  Haystack not installed")


def demo_get_adapter():
    """Demo: Using get_adapter() helper."""
    print("\n" + "="*60)
    print("DEMO: get_adapter() Helper")
    print("="*60)
    
    frameworks = ["langchain", "opencode", "cline", "task"]
    
    for fw in frameworks:
        try:
            memory = get_adapter(fw, base_path=f"./demo_memory/{fw}")
            memory.remember(f"Test for {fw}", importance=0.5)
            results = memory.recall(fw, limit=3)
            print(f"  {fw}: Stored and retrieved {len(results)} items")
            
            # Clean up
            import shutil
            shutil.rmtree(f"./demo_memory/{fw}", ignore_errors=True)
            
        except (ImportError, ValueError) as e:
            print(f"  {fw}: {e}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("AGENT MEMORY ADAPTER DEMOS")
    print("="*70)
    
    # List available adapters
    print("\nAvailable Adapters:")
    adapters = list_adapters()
    for name, info in adapters.items():
        status = "✓" if info["available"] else "✗"
        print(f"  {status} {name}")
    
    # Run demos
    demo_base()
    demo_code_agents()
    demo_langchain()
    demo_openai_agents()
    demo_crewai()
    demo_autogen()
    demo_semantic_kernel()
    demo_llamaindex()
    demo_agno()
    demo_pydantic_ai()
    demo_haystack()
    demo_get_adapter()
    
    # Clean up all demo memory
    import shutil
    shutil.rmtree("./demo_memory", ignore_errors=True)
    
    print("\n" + "="*70)
    print("DEMOS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()