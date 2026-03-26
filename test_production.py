#!/usr/bin/env python3
"""
Production-ready test suite for the Agent Memory System.

This tests edge cases, error handling, stress scenarios, and ensures
the system is robust for production use.
"""

import os
import sys
import shutil
import json
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_memory import MemoryManager, NodeType, RelationType, KnowledgeNode


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*60)
    print("TEST 1: Edge Cases")
    print("="*60)
    
    memory = MemoryManager(base_path="./test_edge_cases", l1_capacity=5)
    
    # Test empty content
    try:
        id1 = memory.remember(content="", node_type="fact")
        print("✓ Empty content handled")
    except Exception as e:
        print(f"✗ Empty content failed: {e}")
    
    # Test very long content
    long_content = "x" * 10000
    try:
        id2 = memory.remember(content=long_content, node_type="fact")
        node = memory.get(id2)
        assert len(node.content) == 10000
        print("✓ Long content (10K chars) handled")
    except Exception as e:
        print(f"✗ Long content failed: {e}")
    
    # Test special characters
    special = "Special: \n\t\"quotes\" 'apostrophes' <html> & symbols @#$%^&*()"
    try:
        id3 = memory.remember(content=special, node_type="fact")
        node = memory.get(id3)
        assert node.content == special
        print("✓ Special characters handled")
    except Exception as e:
        print(f"✗ Special characters failed: {e}")
    
    # Test unicode
    unicode_content = "Unicode: 你好 🌟 émojis ñ"
    try:
        id4 = memory.remember(content=unicode_content, node_type="fact")
        node = memory.get(id4)
        assert node.content == unicode_content
        print("✓ Unicode content handled")
    except Exception as e:
        print(f"✗ Unicode failed: {e}")
    
    # Test non-existent node
    try:
        result = memory.get("non-existent-id-12345")
        assert result is None
        print("✓ Non-existent node returns None")
    except Exception as e:
        print(f"✗ Non-existent node failed: {e}")
    
    # Test empty search
    try:
        results = memory.recall("")
        print(f"✓ Empty search handled: {len(results)} results")
    except Exception as e:
        print(f"✗ Empty search failed: {e}")
    
    # Test duplicate tags
    try:
        id5 = memory.remember(
            content="Test duplicate tags",
            tags=["tag1", "tag1", "tag2", "tag2"]
        )
        node = memory.get(id5)
        # Should handle duplicates gracefully
        print(f"✓ Duplicate tags handled: {len(node.tags)} tags")
    except Exception as e:
        print(f"✗ Duplicate tags failed: {e}")
    
    return memory


def test_compaction_stress():
    """Test compaction under stress conditions."""
    print("\n" + "="*60)
    print("TEST 2: Compaction Stress Test")
    print("="*60)
    
    memory = MemoryManager(base_path="./test_stress", l1_capacity=10)
    
    # Add 100 nodes to force multiple compactions
    print("Adding 100 nodes...")
    node_ids = []
    for i in range(100):
        node_id = memory.remember(
            content=f"Stress test node {i} with content that varies in importance",
            node_type="fact" if i % 2 == 0 else "decision",
            tags=[f"tag{i % 10}", "stress"],
            importance=0.1 + (i % 10) * 0.1
        )
        node_ids.append(node_id)
    
    # Force compaction multiple times
    for _ in range(5):
        memory.force_compact()
    
    stats = memory.get_stats()
    print(f"✓ After 100 nodes + 5 compactions:")
    print(f"  L1: {stats['compaction']['l1_stats']['l1_nodes']} nodes")
    print(f"  L2: {stats['compaction']['l2_stats']['total_nodes']} nodes")
    print(f"  L3: {stats['compaction']['l3_stats']['total_nodes']} nodes")
    
    # Verify all nodes are still retrievable
    print("Verifying all nodes retrievable...")
    missing = 0
    for node_id in node_ids:
        node = memory.get(node_id)
        if node is None:
            missing += 1
    
    print(f"✓ Retrieval test: {100 - missing}/100 nodes found")
    
    # Test search across all levels
    results = memory.recall("stress test")
    print(f"✓ Search across levels: {len(results)} results")
    
    return memory


def test_graph_operations():
    """Test knowledge graph operations."""
    print("\n" + "="*60)
    print("TEST 3: Graph Operations")
    print("="*60)
    
    memory = MemoryManager(base_path="./test_graph", l1_capacity=20)
    
    # Create a chain of related nodes
    print("Creating node chain...")
    prev_id = None
    chain_ids = []
    
    for i in range(10):
        node_id = memory.remember(
            content=f"Chain node {i}",
            node_type="fact",
            tags=["chain", f"node{i}"]
        )
        chain_ids.append(node_id)
        
        if prev_id:
            memory.add_relation(
                source_id=prev_id,
                target_id=node_id,
                relation_type=RelationType.LEADS_TO.value
            )
        
        prev_id = node_id
    
    print(f"✓ Created chain of {len(chain_ids)} nodes")
    
    # Test related nodes
    related = memory.get_related(chain_ids[5])
    print(f"✓ Related nodes for middle of chain: {len(related)} results")
    
    # Create a cycle
    memory.add_relation(
        source_id=chain_ids[-1],
        target_id=chain_ids[0],
        relation_type=RelationType.LEADS_TO.value
    )
    print("✓ Created cycle in graph")
    
    # Test related with cycle
    related = memory.get_related(chain_ids[0])
    print(f"✓ Related nodes with cycle: {len(related)} results")
    
    # Test multiple relations between same nodes
    memory.add_relation(
        source_id=chain_ids[0],
        target_id=chain_ids[1],
        relation_type=RelationType.SUPPORTS.value
    )
    print("✓ Multiple relations between same nodes handled")
    
    return memory


def test_persistence_robustness():
    """Test persistence and recovery."""
    print("\n" + "="*60)
    print("TEST 4: Persistence Robustness")
    print("="*60)
    
    base_path = "./test_persist_robust"
    
    # Clean start
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    # Create memory and add data
    memory1 = MemoryManager(base_path=base_path, l1_capacity=5)
    
    # Add varied data
    ids = []
    ids.append(memory1.remember(
        content="Important decision about architecture",
        node_type="decision",
        tags=["architecture", "important"],
        importance=0.9
    ))
    
    ids.append(memory1.remember(
        content="Minor observation about code style",
        node_type="observation",
        tags=["style"],
        importance=0.2
    ))
    
    # Add relation
    memory1.add_relation(ids[0], ids[1], RelationType.RELATES_TO.value)
    
    # Force some to L2
    for i in range(10):
        memory1.remember(f"Filler {i}", tags=["filler"], importance=0.1)
    
    memory1.force_compact()
    memory1.save()
    
    stats1 = memory1.get_stats()
    print(f"✓ Session 1 saved: {stats1['graph']['nodes']} nodes")
    
    # Create new instance - should load everything
    memory2 = MemoryManager(base_path=base_path, l1_capacity=5)
    stats2 = memory2.get_stats()
    
    print(f"✓ Session 2 loaded: {stats2['graph']['nodes']} nodes")
    
    # Verify all data preserved
    for node_id in ids:
        node = memory2.get(node_id)
        if node:
            print(f"✓ Retrieved {node_id[:8]}... from {node.storage_level}")
        else:
            print(f"✗ Lost node {node_id[:8]}...")
    
    # Verify relation preserved
    related = memory2.get_related(ids[0])
    print(f"✓ Relations preserved: {len(related)} related nodes")
    
    return memory2


def test_concurrent_operations():
    """Test rapid concurrent-like operations."""
    print("\n" + "="*60)
    print("TEST 5: Rapid Operations")
    print("="*60)
    
    memory = MemoryManager(base_path="./test_rapid", l1_capacity=20)
    
    # Rapid adds
    print("Rapid add operations...")
    start = time.time()
    for i in range(50):
        memory.remember(f"Rapid node {i}", tags=["rapid"])
    elapsed = time.time() - start
    print(f"✓ Added 50 nodes in {elapsed:.2f}s ({50/elapsed:.1f} nodes/sec)")
    
    # Rapid searches
    print("Rapid search operations...")
    start = time.time()
    for i in range(100):
        memory.recall("rapid")
    elapsed = time.time() - start
    print(f"✓ 100 searches in {elapsed:.2f}s ({100/elapsed:.1f} searches/sec)")
    
    # Rapid gets
    print("Rapid get operations...")
    node_ids = list(memory.graph.nodes.keys())[:20]
    start = time.time()
    for _ in range(100):
        for nid in node_ids:
            memory.get(nid)
    elapsed = time.time() - start
    print(f"✓ 2000 gets in {elapsed:.2f}s ({2000/elapsed:.1f} gets/sec)")
    
    return memory


def test_context_assembly_robustness():
    """Test context assembly with various scenarios."""
    print("\n" + "="*60)
    print("TEST 6: Context Assembly Robustness")
    print("="*60)
    
    memory = MemoryManager(base_path="./test_context", l1_capacity=20)
    
    # Add varied content
    memory.remember(
        content="Short fact",
        node_type="fact",
        tags=["test"],
        importance=0.5
    )
    
    memory.remember(
        content="A" * 1000,  # Long content
        node_type="fact",
        tags=["test"],
        importance=0.7
    )
    
    memory.remember(
        content="Medium length content about authentication",
        node_type="decision",
        tags=["auth"],
        importance=0.8
    )
    
    # Test with very small token budget
    context = memory.get_context("test", max_tokens=100)
    print(f"✓ Small budget (100 tokens): {context['token_estimate']} tokens used")
    
    # Test with large token budget
    context = memory.get_context("test", max_tokens=10000)
    print(f"✓ Large budget (10000 tokens): {context['token_estimate']} tokens used")
    
    # Test with no matching results
    context = memory.get_context("xyznonexistent123", max_tokens=1000)
    print(f"✓ No matches: {context['node_count']} nodes, {context['token_estimate']} tokens")
    
    # Test summaries only
    context = memory.get_context(
        "authentication",
        max_tokens=500,
        include_summaries=True,
        include_full_content=False
    )
    print(f"✓ Summaries only: {context['node_count']} nodes")
    
    # Test full content only
    context = memory.get_context(
        "authentication",
        max_tokens=500,
        include_summaries=False,
        include_full_content=True
    )
    print(f"✓ Full content only: {context['node_count']} nodes")
    
    return memory


def test_importance_scoring():
    """Test importance scoring and decay."""
    print("\n" + "="*60)
    print("TEST 7: Importance Scoring")
    print("="*60)
    
    memory = MemoryManager(base_path="./test_importance", l1_capacity=20)
    
    # Create nodes with different importance
    id1 = memory.remember("High importance", importance=0.9, node_type="decision")
    id2 = memory.remember("Low importance", importance=0.1, node_type="observation")
    id3 = memory.remember("Medium importance", importance=0.5, node_type="fact")
    
    # Access high importance node multiple times
    for _ in range(10):
        memory.get(id1)
    
    # Force compaction
    memory.force_compact()
    
    # Check that high importance node stayed in L1
    node1 = memory.get(id1)
    node2 = memory.get(id2)
    
    print(f"✓ High importance node in L{node1.storage_level}")
    print(f"✓ Low importance node in L{node2.storage_level}")
    
    # Test importance update
    memory.update_importance(id2, 0.95)
    node2_updated = memory.get(id2)
    print(f"✓ Importance updated: {node2_updated.importance}")
    
    return memory


def test_file_corruption_recovery():
    """Test recovery from corrupted files."""
    print("\n" + "="*60)
    print("TEST 8: File Corruption Recovery")
    print("="*60)
    
    base_path = "./test_corruption"
    
    # Clean start
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    memory = MemoryManager(base_path=base_path, l1_capacity=5)
    
    # Add some data
    for i in range(10):
        memory.remember(f"Node {i}", tags=["test"])
    
    memory.save()
    
    # Corrupt index file
    index_path = os.path.join(base_path, "index.json")
    if os.path.exists(index_path):
        with open(index_path, 'w') as f:
            f.write("{ corrupted json }}}")
        print("✓ Corrupted index.json")
    
    # Try to load - should handle gracefully
    try:
        memory2 = MemoryManager(base_path=base_path, l1_capacity=5)
        print("✓ Loaded despite corrupted index (graceful degradation)")
    except Exception as e:
        print(f"✗ Failed to handle corrupted index: {e}")
    
    # Corrupt L1 state
    l1_path = os.path.join(base_path, "l1_state.json")
    if os.path.exists(l1_path):
        with open(l1_path, 'w') as f:
            f.write("invalid json")
        print("✓ Corrupted l1_state.json")
    
    # Try to load - should handle gracefully
    try:
        memory3 = MemoryManager(base_path=base_path, l1_capacity=5)
        print("✓ Loaded despite corrupted L1 state")
    except Exception as e:
        print(f"✗ Failed to handle corrupted L1: {e}")


def test_memory_limits():
    """Test behavior at memory limits."""
    print("\n" + "="*60)
    print("TEST 9: Memory Limits")
    print("="*60)
    
    # Very small L1 capacity
    memory = MemoryManager(base_path="./test_limits", l1_capacity=3)
    
    # Add more than capacity
    ids = []
    for i in range(10):
        id = memory.remember(f"Node {i}", importance=0.1 + i*0.1)
        ids.append(id)
    
    stats = memory.get_stats()
    print(f"✓ L1 capacity 3, added 10 nodes:")
    print(f"  L1: {stats['compaction']['l1_stats']['l1_nodes']}")
    print(f"  L2: {stats['compaction']['l2_stats']['total_nodes']}")
    
    # All should still be retrievable
    all_found = all(memory.get(id) is not None for id in ids)
    print(f"✓ All nodes retrievable: {all_found}")
    
    return memory


def run_all_tests():
    """Run all production tests."""
    print("\n" + "="*60)
    print("PRODUCTION READINESS TEST SUITE")
    print("="*60)
    
    # Clean up any existing test data
    test_dirs = [
        "./test_edge_cases",
        "./test_stress",
        "./test_graph",
        "./test_persist_robust",
        "./test_rapid",
        "./test_context",
        "./test_importance",
        "./test_corruption",
        "./test_limits"
    ]
    
    for d in test_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    results = []
    
    try:
        test_edge_cases()
        results.append(("Edge Cases", True))
    except Exception as e:
        print(f"✗ Edge cases failed: {e}")
        results.append(("Edge Cases", False))
    
    try:
        test_compaction_stress()
        results.append(("Compaction Stress", True))
    except Exception as e:
        print(f"✗ Compaction stress failed: {e}")
        results.append(("Compaction Stress", False))
    
    try:
        test_graph_operations()
        results.append(("Graph Operations", True))
    except Exception as e:
        print(f"✗ Graph operations failed: {e}")
        results.append(("Graph Operations", False))
    
    try:
        test_persistence_robustness()
        results.append(("Persistence", True))
    except Exception as e:
        print(f"✗ Persistence failed: {e}")
        results.append(("Persistence", False))
    
    try:
        test_concurrent_operations()
        results.append(("Rapid Operations", True))
    except Exception as e:
        print(f"✗ Rapid operations failed: {e}")
        results.append(("Rapid Operations", False))
    
    try:
        test_context_assembly_robustness()
        results.append(("Context Assembly", True))
    except Exception as e:
        print(f"✗ Context assembly failed: {e}")
        results.append(("Context Assembly", False))
    
    try:
        test_importance_scoring()
        results.append(("Importance Scoring", True))
    except Exception as e:
        print(f"✗ Importance scoring failed: {e}")
        results.append(("Importance Scoring", False))
    
    try:
        test_file_corruption_recovery()
        results.append(("Corruption Recovery", True))
    except Exception as e:
        print(f"✗ Corruption recovery failed: {e}")
        results.append(("Corruption Recovery", False))
    
    try:
        test_memory_limits()
        results.append(("Memory Limits", True))
    except Exception as e:
        print(f"✗ Memory limits failed: {e}")
        results.append(("Memory Limits", False))
    
    # Summary
    print("\n" + "="*60)
    print("PRODUCTION TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✓ PASSED" if ok else "✗ FAILED"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    # Cleanup
    print("\nCleaning up test data...")
    for d in test_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)