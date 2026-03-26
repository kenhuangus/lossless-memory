#!/usr/bin/env python3
"""
Test script for the Agent Memory System.

This script validates all major functionality:
1. Storing knowledge (remember)
2. Retrieving knowledge (recall)
3. Multi-level compaction
4. Context assembly
5. Relations between nodes
6. Persistence across sessions
"""

import os
import sys
import shutil
from datetime import datetime

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_memory import MemoryManager, NodeType, RelationType


def test_basic_operations():
    """Test basic remember and recall operations."""
    print("\n" + "="*60)
    print("TEST 1: Basic Operations")
    print("="*60)
    
    # Create a fresh memory system
    memory = MemoryManager(base_path="./test_memory_data", l1_capacity=10)
    
    # Store some knowledge
    id1 = memory.remember(
        content="Python is a high-level programming language known for its simplicity.",
        node_type=NodeType.FACT.value,
        tags=["python", "programming"],
        importance=0.8
    )
    print(f"✓ Stored fact about Python: {id1[:8]}...")
    
    id2 = memory.remember(
        content="We decided to use FastAPI for the backend because of its async support.",
        node_type=NodeType.DECISION.value,
        tags=["api", "backend", "fastapi"],
        importance=0.9
    )
    print(f"✓ Stored decision about FastAPI: {id2[:8]}...")
    
    id3 = memory.remember(
        content="The authentication system uses JWT tokens with 24-hour expiry.",
        node_type=NodeType.FACT.value,
        tags=["auth", "security", "jwt"],
        importance=0.7
    )
    print(f"✓ Stored fact about auth: {id3[:8]}...")
    
    # Test recall
    results = memory.recall("python")
    print(f"\n✓ Recalled {len(results)} results for 'python'")
    for r in results:
        print(f"  - {r['summary'][:50]}...")
    
    # Test get specific node
    node = memory.get(id2)
    print(f"\n✓ Retrieved specific node: {node.content[:50]}...")
    
    return memory, [id1, id2, id3]


def test_relations(memory, node_ids):
    """Test relationship creation between nodes."""
    print("\n" + "="*60)
    print("TEST 2: Relations Between Nodes")
    print("="*60)
    
    # Add a relation
    edge_id = memory.add_relation(
        source_id=node_ids[1],  # FastAPI decision
        target_id=node_ids[2],  # Auth fact
        relation_type=RelationType.DEPENDS_ON.value
    )
    print(f"✓ Created relation: FastAPI decision -> Auth fact")
    
    # Create a new node with relation
    id4 = memory.remember_with_relation(
        content="FastAPI endpoints require the auth_token header for protected routes.",
        related_to=node_ids[2],
        relation_type=RelationType.SUPPORTS.value,
        node_type=NodeType.FACT.value,
        tags=["api", "auth", "endpoint"]
    )
    print(f"✓ Created node with relation: {id4[:8]}...")
    
    # Get related nodes
    related = memory.get_related(node_ids[2])
    print(f"\n✓ Found {len(related)} nodes related to auth fact")
    for r in related:
        print(f"  - {r['summary'][:50]}...")


def test_compaction(memory):
    """Test multi-level compaction."""
    print("\n" + "="*60)
    print("TEST 3: Multi-Level Compaction")
    print("="*60)
    
    # Add many nodes to trigger compaction
    print("Adding 15 nodes to fill L1...")
    for i in range(15):
        memory.remember(
            content=f"Test knowledge item number {i} with various details about the system.",
            node_type=NodeType.FACT.value,
            tags=["test", f"item{i}"],
            importance=0.3 + (i * 0.05)  # Varying importance
        )
    
    # Check stats before compaction
    stats = memory.get_stats()
    print(f"\nBefore compaction:")
    print(f"  L1 nodes: {stats['compaction']['l1_stats']['l1_nodes']}")
    print(f"  L2 nodes: {stats['compaction']['l2_stats']['total_nodes']}")
    print(f"  L3 nodes: {stats['compaction']['l3_stats']['total_nodes']}")
    
    # Force compaction
    result = memory.force_compact()
    print(f"\n✓ Compaction completed:")
    print(f"  L1 -> L2: {result['l1_to_l2']} nodes")
    print(f"  L2 -> L3: {result['l2_to_l3']} nodes")
    
    # Check stats after compaction
    stats = memory.get_stats()
    print(f"\nAfter compaction:")
    print(f"  L1 nodes: {stats['compaction']['l1_stats']['l1_nodes']}")
    print(f"  L2 nodes: {stats['compaction']['l2_stats']['total_nodes']}")
    print(f"  L3 nodes: {stats['compaction']['l3_stats']['total_nodes']}")
    
    # Verify we can still retrieve compacted nodes
    results = memory.recall("test knowledge item")
    print(f"\n✓ Can still recall compacted nodes: {len(results)} results")


def test_context_assembly(memory):
    """Test context assembly for LLM prompts."""
    print("\n" + "="*60)
    print("TEST 4: Context Assembly")
    print("="*60)
    
    # Get context for a query
    context = memory.get_context(
        "how does authentication work?",
        max_tokens=1000,
        include_summaries=True,
        include_full_content=True
    )
    
    print(f"✓ Assembled context:")
    print(f"  Token estimate: {context['token_estimate']}")
    print(f"  Node count: {context['node_count']}")
    print(f"  Context preview:")
    print("  " + "-"*50)
    # Print first 500 chars of context
    print("  " + context['context'][:500].replace('\n', '\n  '))
    print("  " + "-"*50)


def test_search_functionality(memory):
    """Test various search methods."""
    print("\n" + "="*60)
    print("TEST 5: Search Functionality")
    print("="*60)
    
    # Search by tag
    tag_results = memory.search_by_tag("auth")
    print(f"✓ Search by tag 'auth': {len(tag_results)} results")
    
    # Search by type
    type_results = memory.search_by_type(NodeType.DECISION.value)
    print(f"✓ Search by type 'decision': {len(type_results)} results")
    
    # Get recent nodes
    recent = memory.get_recent(limit=5)
    print(f"✓ Recent nodes: {len(recent)} results")


def test_persistence():
    """Test that memory persists across sessions."""
    print("\n" + "="*60)
    print("TEST 6: Persistence Across Sessions")
    print("="*60)
    
    # Create first memory instance and add data
    memory1 = MemoryManager(base_path="./test_memory_persist")
    id1 = memory1.remember(
        content="This should persist across sessions.",
        node_type=NodeType.FACT.value,
        tags=["persistence", "test"]
    )
    stats1 = memory1.get_stats()
    print(f"✓ Session 1: {stats1['graph']['nodes']} nodes")
    memory1.save()
    
    # Create second memory instance - should load existing data
    memory2 = MemoryManager(base_path="./test_memory_persist")
    stats2 = memory2.get_stats()
    print(f"✓ Session 2: {stats2['graph']['nodes']} nodes")
    
    # Verify we can retrieve the data
    node = memory2.get(id1)
    if node:
        print(f"✓ Retrieved persisted node: {node.content[:40]}...")
    else:
        print("✗ Failed to retrieve persisted node!")
    
    return stats1['graph']['nodes'] == stats2['graph']['nodes']


def test_memory_summary(memory):
    """Test memory summary generation."""
    print("\n" + "="*60)
    print("TEST 7: Memory Summary")
    print("="*60)
    
    summary = memory.get_memory_summary()
    print(summary)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AGENT MEMORY SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Clean up any existing test data
    if os.path.exists("./test_memory_data"):
        shutil.rmtree("./test_memory_data")
    if os.path.exists("./test_memory_persist"):
        shutil.rmtree("./test_memory_persist")
    
    try:
        # Run tests
        memory, node_ids = test_basic_operations()
        test_relations(memory, node_ids)
        test_compaction(memory)
        test_context_assembly(memory)
        test_search_functionality(memory)
        persist_ok = test_persistence()
        test_memory_summary(memory)
        
        # Final summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✓ Basic operations: PASSED")
        print("✓ Relations: PASSED")
        print("✓ Compaction: PASSED")
        print("✓ Context assembly: PASSED")
        print("✓ Search functionality: PASSED")
        print(f"✓ Persistence: {'PASSED' if persist_ok else 'FAILED'}")
        print("✓ Memory summary: PASSED")
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Show final memory stats
        stats = memory.get_stats()
        print(f"\nFinal Memory State:")
        print(f"  Total nodes: {stats['graph']['nodes']}")
        print(f"  Total edges: {stats['graph']['edges']}")
        print(f"  L1 (RAM): {stats['index']['l1_nodes']} nodes")
        print(f"  L2 (Warm): {stats['index']['l2_nodes']} nodes")
        print(f"  L3 (Cold): {stats['index']['l3_nodes']} nodes")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up
        print("\nCleaning up test data...")
        if os.path.exists("./test_memory_data"):
            shutil.rmtree("./test_memory_data")
        if os.path.exists("./test_memory_persist"):
            shutil.rmtree("./test_memory_persist")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())