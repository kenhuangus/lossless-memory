#!/usr/bin/env python3
"""
Demo: Google ADK Integration with Agent Memory System.

This script demonstrates how to use the Agent Memory System
with Google ADK for long-term memory capabilities.
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_memory import MemoryManager
from agent_memory.adk import create_memory_tool, MemoryEnhancedAgent, MemoryCallback


def demo_memory_tool():
    """Demonstrate using MemoryTool with ADK."""
    print("\n" + "="*60)
    print("DEMO 1: MemoryTool with ADK")
    print("="*60)
    
    # Initialize memory
    memory = MemoryManager(base_path="./demo_adk_memory")
    
    # Create memory tool
    tool = create_memory_tool(memory=memory)
    
    # Store some knowledge (call tool with action parameter)
    result = tool.func(
        action="remember",
        content="The API uses JWT tokens with 24-hour expiry",
        node_type="fact",
        tags=["api", "auth"],
        importance=0.8
    )
    print(f"✓ Stored: {result}")
    
    # Search for knowledge
    result = tool.func(action="recall", query="authentication", limit=5)
    print(f"✓ Recalled: {result['count']} results")
    
    # Get context for LLM
    result = tool.func(action="get_context", query="how does auth work?", max_tokens=1000)
    print(f"✓ Context: {result['token_estimate']} tokens")
    print(f"\n{result['context']}")
    
    # Get stats
    result = tool.func(action="get_stats")
    print(f"✓ Stats: {result['stats']['graph']['nodes']} nodes")
    
    # Clean up
    memory.clear()
    print("\n✓ Demo 1 complete!")


def demo_memory_callback():
    """Demonstrate using MemoryCallback with ADK."""
    print("\n" + "="*60)
    print("DEMO 2: MemoryCallback for Automatic Memory")
    print("="*60)
    
    # Initialize callback
    callback = MemoryCallback(memory_path="./demo_adk_callback")
    
    # Simulate agent interactions
    print("\nSimulating agent interactions...")
    
    # User asks a question
    context = callback.on_message("What database should we use for the project?")
    print(f"✓ Retrieved context: {context}")
    
    # Agent responds with a decision
    callback.on_response(
        message="What database should we use for the project?",
        response="Based on our requirements, I recommend PostgreSQL. It provides excellent ACID compliance, JSON support, and scalability. Therefore, we should proceed with PostgreSQL for our backend."
    )
    print("✓ Stored decision automatically")
    
    # Another interaction
    context = callback.on_message("How do we handle authentication?")
    print(f"✓ Retrieved context: {context}")
    
    callback.on_response(
        message="How do we handle authentication?",
        response="We will implement JWT-based authentication with refresh tokens. The solution involves using PyJWT for token generation and validation."
    )
    print("✓ Stored solution automatically")
    
    # Simulate an error
    callback.on_error(
        error=Exception("Connection timeout"),
        context="Database connection attempt"
    )
    print("✓ Stored error automatically")
    
    # Check what was stored
    results = callback.recall("database", limit=5)
    print(f"\n✓ Found {len(results)} results about 'database':")
    for r in results:
        print(f"  - {r['summary'][:60]}...")
    
    # Get stats
    stats = callback.get_stats()
    print(f"\n✓ Total nodes stored: {stats['graph']['nodes']}")
    
    # Save and clean up
    callback.save()
    callback.clear()
    print("\n✓ Demo 2 complete!")


def demo_memory_enhanced_agent():
    """Demonstrate using MemoryEnhancedAgent."""
    print("\n" + "="*60)
    print("DEMO 3: MemoryEnhancedAgent Wrapper")
    print("="*60)
    
    # Create a mock agent for demonstration
    class MockAgent:
        """Mock ADK agent for demonstration."""
        
        async def run(self, message: str, context=None, **kwargs):
            """Simulate agent response."""
            # Check if memory context was injected
            if "Relevant Knowledge from Memory" in message:
                return "Based on my memory, I can provide a more informed answer..."
            else:
                return "I don't have prior context, but here's my response..."
    
    # Create base agent
    base_agent = MockAgent()
    
    # Wrap with memory
    agent = MemoryEnhancedAgent(
        agent=base_agent,
        memory_path="./demo_adk_enhanced",
        memory_capacity=50,
        auto_store=True,
        auto_retrieve=True
    )
    
    print("\nSimulating conversation with memory-enhanced agent...")
    
    # First interaction - no memory yet
    async def run_demo():
        response = await agent.run("What's the best programming language for web APIs?")
        print(f"\n1. User: What's the best programming language for web APIs?")
        print(f"   Agent: {response}")
        
        # Store some knowledge manually
        agent.remember_manually(
            content="We decided to use Python with FastAPI for our backend",
            node_type="decision",
            tags=["architecture", "backend"],
            importance=0.9
        )
        print("\n✓ Manually stored decision about FastAPI")
        
        # Second interaction - should have memory context
        response = await agent.run("What framework should we use for the API?")
        print(f"\n2. User: What framework should we use for the API?")
        print(f"   Agent: {response}")
        
        # Third interaction
        response = await agent.run("Tell me about our backend architecture")
        print(f"\n3. User: Tell me about our backend architecture")
        print(f"   Agent: {response}")
    
    # Run the async demo
    asyncio.run(run_demo())
    
    # Show conversation history
    history = agent.get_conversation_history()
    print(f"\n✓ Conversation history: {len(history)} messages")
    
    # Show memory stats
    stats = agent.get_memory_stats()
    print(f"✓ Memory nodes: {stats['graph']['nodes']}")
    
    # Export memory
    agent.export_memory("demo_memory_export.md")
    print("✓ Exported memory to demo_memory_export.md")
    
    # Clean up
    agent.clear_memory()
    print("\n✓ Demo 3 complete!")


def demo_advanced_usage():
    """Demonstrate advanced usage patterns."""
    print("\n" + "="*60)
    print("DEMO 4: Advanced Usage Patterns")
    print("="*60)
    
    memory = MemoryManager(base_path="./demo_adk_advanced")
    
    # Store related knowledge with relationships
    print("\nCreating knowledge graph with relationships...")
    
    # Store a decision
    decision_id = memory.remember(
        content="We decided to use PostgreSQL as our primary database",
        node_type="decision",
        tags=["database", "architecture"],
        importance=0.9
    )
    print(f"✓ Stored decision: {decision_id[:8]}...")
    
    # Store related facts
    fact1_id = memory.remember(
        content="PostgreSQL supports JSONB for flexible schema",
        node_type="fact",
        tags=["database", "postgresql"],
        importance=0.7
    )
    
    fact2_id = memory.remember(
        content="PostgreSQL requires Python psycopg2 driver",
        node_type="fact",
        tags=["database", "postgresql", "python"],
        importance=0.6
    )
    
    # Create relationships
    memory.add_relation(decision_id, fact1_id, "supports")
    memory.add_relation(decision_id, fact2_id, "depends_on")
    print("✓ Created relationships between nodes")
    
    # Store an error and solution
    error_id = memory.remember(
        content="Connection pool exhausted under high load",
        node_type="error",
        tags=["database", "performance"],
        importance=0.95
    )
    
    solution_id = memory.remember(
        content="Increased pool size to 20 connections and added connection recycling",
        node_type="solution",
        tags=["database", "performance"],
        importance=0.9
    )
    
    memory.add_relation(error_id, solution_id, "solves")
    print("✓ Created error-solution relationship")
    
    # Search by tag
    db_nodes = memory.search_by_tag("database")
    print(f"\n✓ Found {len(db_nodes)} nodes with 'database' tag")
    
    # Get related nodes
    related = memory.get_related(decision_id)
    print(f"✓ Found {len(related)} nodes related to decision")
    
    # Get context with full content
    context = memory.get_context(
        "database architecture",
        max_tokens=2000,
        include_summaries=True,
        include_full_content=True
    )
    print(f"\n✓ Context assembled: {context['node_count']} nodes, {context['token_estimate']} tokens")
    print(f"\n{context['context'][:500]}...")
    
    # Force compaction
    stats = memory.force_compact()
    print(f"\n✓ Compaction: L1→L2: {stats['l1_to_l2']}, L2→L3: {stats['l2_to_l3']}")
    
    # Export
    memory.export_markdown("demo_advanced_export.md")
    print("✓ Exported to demo_advanced_export.md")
    
    # Clean up
    memory.clear()
    print("\n✓ Demo 4 complete!")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("AGENT MEMORY + GOOGLE ADK INTEGRATION DEMOS")
    print("="*60)
    
    # Clean up any existing demo data
    import shutil
    demo_dirs = [
        "./demo_adk_memory",
        "./demo_adk_callback",
        "./demo_adk_enhanced",
        "./demo_adk_advanced"
    ]
    for d in demo_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    try:
        # Run demos
        demo_memory_tool()
        demo_memory_callback()
        demo_memory_enhanced_agent()
        demo_advanced_usage()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        print("\nCleaning up demo data...")
        for d in demo_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
        
        # Clean up export files
        for f in ["demo_memory_export.md", "demo_advanced_export.md"]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    main()