#!/usr/bin/env python3
"""
Test script for the flexible N-level memory architecture.

This script validates:
1. Different preset configurations (2-6 levels)
2. Custom configurations
3. Preset listing and selection
4. Backward compatibility with legacy parameters
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_memory import (
    MemoryManager,
    get_preset,
    list_presets,
    create_custom_config,
    NodeType
)


def test_preset_listing():
    """Test listing available presets."""
    print("\n" + "="*60)
    print("TEST 1: List Available Presets")
    print("="*60)
    
    presets = MemoryManager.list_available_presets()
    
    print(f"Found {len(presets)} presets:")
    for name, description in presets.items():
        print(f"  - {name}: {description}")
    
    assert len(presets) >= 5, "Should have at least 5 presets"
    assert "chatbot" in presets
    assert "assistant" in presets
    assert "enterprise" in presets
    assert "regulatory" in presets
    assert "research" in presets
    
    print("✓ All presets available")


def test_chatbot_preset():
    """Test 2-level chatbot configuration."""
    print("\n" + "="*60)
    print("TEST 2: Chatbot Preset (2 levels)")
    print("="*60)
    
    memory = MemoryManager(
        base_path="./test_chatbot",
        memory_preset="chatbot"
    )
    
    # Verify configuration
    config = memory.get_config()
    print(f"Configuration: {config.name}")
    print(f"Number of levels: {config.num_levels}")
    
    assert config.num_levels == 2, "Chatbot should have 2 levels"
    assert config.levels[0].name == "hot"
    assert config.levels[1].name == "cold"
    
    # Test basic operations
    node_id = memory.remember(
        content="How can I help you today?",
        node_type=NodeType.FACT.value,
        tags=["greeting"]
    )
    print(f"✓ Stored node: {node_id[:8]}...")
    
    # Test recall
    results = memory.recall("help")
    print(f"✓ Recalled {len(results)} results")
    
    # Get stats
    stats = memory.get_stats()
    print(f"✓ Config: {stats['config']['name']} ({stats['config']['num_levels']} levels)")
    
    memory.clear()
    print("✓ Chatbot preset test passed")


def test_assistant_preset():
    """Test 3-level assistant configuration (default)."""
    print("\n" + "="*60)
    print("TEST 3: Assistant Preset (3 levels)")
    print("="*60)
    
    memory = MemoryManager(
        base_path="./test_assistant",
        memory_preset="assistant"
    )
    
    config = memory.get_config()
    print(f"Configuration: {config.name}")
    print(f"Number of levels: {config.num_levels}")
    
    assert config.num_levels == 3, "Assistant should have 3 levels"
    assert config.levels[0].name == "hot"
    assert config.levels[1].name == "warm"
    assert config.levels[2].name == "cold"
    
    # Test operations
    for i in range(15):
        memory.remember(
            content=f"Knowledge item {i} for testing",
            node_type=NodeType.FACT.value,
            tags=["test"],
            importance=0.3 + i * 0.05
        )
    
    print(f"✓ Stored 15 nodes")
    
    # Force compaction
    memory.force_compact()
    stats = memory.get_stats()
    print(f"✓ L1: {stats['compaction']['l1_stats']['l1_nodes']} nodes")
    print(f"✓ L2: {stats['compaction']['l2_stats']['total_nodes']} nodes")
    
    memory.clear()
    print("✓ Assistant preset test passed")


def test_enterprise_preset():
    """Test 5-level enterprise configuration."""
    print("\n" + "="*60)
    print("TEST 4: Enterprise Preset (5 levels)")
    print("="*60)
    
    memory = MemoryManager(
        base_path="./test_enterprise",
        memory_preset="enterprise"
    )
    
    config = memory.get_config()
    print(f"Configuration: {config.name}")
    print(f"Number of levels: {config.num_levels}")
    print(f"Levels: {[l.name for l in config.levels]}")
    
    assert config.num_levels == 5, "Enterprise should have 5 levels"
    level_names = [l.name for l in config.levels]
    assert level_names == ["hot", "warm", "cool", "cold", "archive"]
    
    # Test operations
    for i in range(30):
        memory.remember(
            content=f"Enterprise knowledge {i}",
            node_type=NodeType.FACT.value if i % 2 == 0 else NodeType.DECISION.value,
            tags=["enterprise", f"item{i}"],
            importance=0.2 + i * 0.03
        )
    
    print(f"✓ Stored 30 nodes")
    
    # Multiple compactions
    for _ in range(3):
        memory.force_compact()
    
    stats = memory.get_stats()
    print(f"✓ After compaction:")
    print(f"  L1: {stats['compaction']['l1_stats']['l1_nodes']} nodes")
    print(f"  L2: {stats['compaction']['l2_stats']['total_nodes']} nodes")
    print(f"  L3: {stats['compaction']['l3_stats']['total_nodes']} nodes")
    
    memory.clear()
    print("✓ Enterprise preset test passed")


def test_regulatory_preset():
    """Test 6-level regulatory configuration."""
    print("\n" + "="*60)
    print("TEST 5: Regulatory Preset (6 levels)")
    print("="*60)
    
    memory = MemoryManager(
        base_path="./test_regulatory",
        memory_preset="regulatory"
    )
    
    config = memory.get_config()
    print(f"Configuration: {config.name}")
    print(f"Number of levels: {config.num_levels}")
    print(f"Levels: {[l.name for l in config.levels]}")
    
    assert config.num_levels == 6, "Regulatory should have 6 levels"
    level_names = [l.name for l in config.levels]
    assert level_names == ["hot", "warm", "cool", "cold", "archive", "permanent"]
    
    # Verify retention periods
    retention_periods = [l.retention_period for l in config.levels]
    print(f"✓ Retention periods: {retention_periods}")
    
    memory.clear()
    print("✓ Regulatory preset test passed")


def test_custom_config():
    """Test custom configuration."""
    print("\n" + "="*60)
    print("TEST 6: Custom Configuration (4 levels)")
    print("="*60)
    
    # Create custom 4-level config
    config = create_custom_config(
        num_levels=4,
        hot_capacity=200,
        retention_periods=[None, "12h", "7d", "30d"]
    )
    
    memory = MemoryManager(
        base_path="./test_custom",
        config=config
    )
    
    print(f"Configuration: {config.name}")
    print(f"Number of levels: {config.num_levels}")
    print(f"Levels: {[l.name for l in config.levels]}")
    print(f"Hot capacity: {config.levels[0].capacity}")
    
    assert config.num_levels == 4
    assert config.levels[0].capacity == 200
    
    # Test operations
    for i in range(25):
        memory.remember(
            content=f"Custom level test {i}",
            tags=["custom"]
        )
    
    print(f"✓ Stored 25 nodes")
    
    memory.clear()
    print("✓ Custom configuration test passed")


def test_backward_compatibility():
    """Test backward compatibility with legacy parameters."""
    print("\n" + "="*60)
    print("TEST 7: Backward Compatibility")
    print("="*60)
    
    # Use legacy parameters
    memory = MemoryManager(
        base_path="./test_legacy",
        l1_capacity=50,
        auto_compact=True,
        compaction_threshold=0.7
    )
    
    config = memory.get_config()
    print(f"Configuration: {config.name}")
    print(f"L1 capacity: {config.levels[0].capacity}")
    print(f"Auto compact: {config.auto_compact}")
    print(f"Compaction threshold: {config.compaction_threshold}")
    
    # Verify legacy parameters were applied
    assert config.levels[0].capacity == 50
    assert config.auto_compact == True
    assert config.compaction_threshold == 0.7
    
    # Test basic operations
    node_id = memory.remember("Backward compatibility test")
    print(f"✓ Stored node: {node_id[:8]}...")
    
    memory.clear()
    print("✓ Backward compatibility test passed")


def test_research_preset():
    """Test 4-level research configuration."""
    print("\n" + "="*60)
    print("TEST 8: Research Preset (4 levels)")
    print("="*60)
    
    memory = MemoryManager(
        base_path="./test_research",
        memory_preset="research"
    )
    
    config = memory.get_config()
    print(f"Configuration: {config.name}")
    print(f"Number of levels: {config.num_levels}")
    
    assert config.num_levels == 4
    level_names = [l.name for l in config.levels]
    assert level_names == ["hot", "warm", "cool", "cold"]
    
    memory.clear()
    print("✓ Research preset test passed")


def run_all_tests():
    """Run all flexible level tests."""
    print("\n" + "="*60)
    print("FLEXIBLE N-LEVEL ARCHITECTURE TEST SUITE")
    print("="*60)
    
    # Clean up any existing test data
    test_dirs = [
        "./test_chatbot",
        "./test_assistant",
        "./test_enterprise",
        "./test_regulatory",
        "./test_custom",
        "./test_legacy",
        "./test_research"
    ]
    
    for d in test_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    results = []
    
    tests = [
        ("Preset Listing", test_preset_listing),
        ("Chatbot Preset", test_chatbot_preset),
        ("Assistant Preset", test_assistant_preset),
        ("Enterprise Preset", test_enterprise_preset),
        ("Regulatory Preset", test_regulatory_preset),
        ("Custom Config", test_custom_config),
        ("Backward Compatibility", test_backward_compatibility),
        ("Research Preset", test_research_preset)
    ]
    
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
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