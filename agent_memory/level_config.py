"""
Memory Level Configuration - Flexible N-level memory hierarchy.

This module provides configuration for dynamic memory levels,
allowing systems to scale from 2 levels (simple chatbots) to
7+ levels (enterprise compliance systems).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class StorageType(Enum):
    """Types of storage backends."""
    RAM = "ram"                    # In-memory storage
    FILE_INDEXED = "file_indexed"  # File with RAM index
    FILE_COLD = "file_cold"        # Compressed archive files
    DATABASE = "database"          # SQL/NoSQL database
    VECTOR_DB = "vector_db"        # Vector database for semantic search


@dataclass
class LevelSpec:
    """
    Specification for a single memory level.
    
    Attributes:
        name: Human-readable name (e.g., "hot", "warm", "cold")
        level_num: Numeric level (0 = hot/highest priority)
        capacity: Maximum nodes (None = unlimited)
        storage_type: How this level stores data
        importance_threshold: Min importance to stay at this level
        retention_period: How long to keep data (e.g., "7d", "30d", "1y", None=forever)
        auto_promote: Automatically promote on access
        compress: Use compression for file storage
        searchable: Include in search operations
        description: Human-readable description
    """
    name: str
    level_num: int
    capacity: Optional[int]
    storage_type: StorageType
    importance_threshold: float = 0.0
    retention_period: Optional[str] = None
    auto_promote: bool = True
    compress: bool = False
    searchable: bool = True
    description: str = ""


@dataclass
class MemoryConfig:
    """
    Complete memory configuration for N-level hierarchy.
    
    Attributes:
        name: Configuration name
        description: What this config is designed for
        levels: List of level specifications
        auto_compact: Enable automatic compaction
        compaction_threshold: L1 utilization to trigger compaction
        decay_factor: Importance decay per day (0-1)
        default_preset: Default preset name if using presets
    """
    name: str
    description: str
    levels: List[LevelSpec]
    auto_compact: bool = True
    compaction_threshold: float = 0.8
    decay_factor: float = 0.95
    default_preset: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if len(self.levels) < 2:
            raise ValueError("Memory must have at least 2 levels")
        
        # Ensure levels are sorted by level_num
        self.levels.sort(key=lambda x: x.level_num)
        
        # Validate level numbers are sequential
        for i, level in enumerate(self.levels):
            if level.level_num != i:
                raise ValueError(f"Level numbers must be sequential, got {level.level_num} at index {i}")
    
    @property
    def num_levels(self) -> int:
        """Get number of memory levels."""
        return len(self.levels)
    
    @property
    def hot_level(self) -> LevelSpec:
        """Get the hottest (L1) level."""
        return self.levels[0]
    
    @property
    def cold_level(self) -> LevelSpec:
        """Get the coldest (last) level."""
        return self.levels[-1]
    
    def get_level(self, level_num: int) -> LevelSpec:
        """Get level specification by number."""
        if level_num < 0 or level_num >= len(self.levels):
            raise ValueError(f"Invalid level number: {level_num}")
        return self.levels[level_num]
    
    def get_level_by_name(self, name: str) -> Optional[LevelSpec]:
        """Get level specification by name."""
        for level in self.levels:
            if level.name.lower() == name.lower():
                return level
        return None


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def create_chatbot_config() -> MemoryConfig:
    """
    Simple 2-level config for short-lived chatbots.
    
    - L0: Hot RAM (100 nodes, 1 hour retention)
    - L1: Cold file (unlimited, 24 hour retention)
    
    Use case: Customer support bot, FAQ assistant
    """
    return MemoryConfig(
        name="chatbot",
        description="Simple 2-level memory for short conversations (hours)",
        levels=[
            LevelSpec(
                name="hot",
                level_num=0,
                capacity=100,
                storage_type=StorageType.RAM,
                importance_threshold=0.3,
                retention_period="1h",
                auto_promote=True,
                description="Active conversation context"
            ),
            LevelSpec(
                name="cold",
                level_num=1,
                capacity=None,  # Unlimited
                storage_type=StorageType.FILE_COLD,
                importance_threshold=0.0,
                retention_period="24h",
                auto_promote=True,
                compress=True,
                description="Recent conversation history"
            )
        ]
    )


def create_assistant_config() -> MemoryConfig:
    """
    Standard 3-level config for general assistants (current default).
    
    - L0: Hot RAM (100 nodes, immediate access)
    - L1: Warm indexed files (1000s of nodes, 7 day retention)
    - L2: Cold archive (unlimited, 30 day retention)
    
    Use case: Personal assistant, coding helper, research aid
    """
    return MemoryConfig(
        name="assistant",
        description="Standard 3-level memory for daily assistants (days to weeks)",
        levels=[
            LevelSpec(
                name="hot",
                level_num=0,
                capacity=100,
                storage_type=StorageType.RAM,
                importance_threshold=0.4,
                retention_period=None,  # Keep while important
                auto_promote=True,
                description="Active working memory"
            ),
            LevelSpec(
                name="warm",
                level_num=1,
                capacity=None,
                storage_type=StorageType.FILE_INDEXED,
                importance_threshold=0.2,
                retention_period="7d",
                auto_promote=True,
                description="Recently accessed knowledge"
            ),
            LevelSpec(
                name="cold",
                level_num=2,
                capacity=None,
                storage_type=StorageType.FILE_COLD,
                importance_threshold=0.0,
                retention_period="30d",
                auto_promote=True,
                compress=True,
                description="Archived knowledge"
            )
        ]
    )


def create_enterprise_config() -> MemoryConfig:
    """
    5-level config for enterprise systems with longer retention.
    
    - L0: Hot RAM (200 nodes)
    - L1: Warm indexed (today's knowledge)
    - L2: Cool indexed (this week)
    - L3: Cold compressed (this month)
    - L4: Deep archive (this year)
    
    Use case: Project management, customer support systems
    """
    return MemoryConfig(
        name="enterprise",
        description="5-level memory for enterprise systems (weeks to months)",
        levels=[
            LevelSpec(
                name="hot",
                level_num=0,
                capacity=200,
                storage_type=StorageType.RAM,
                importance_threshold=0.5,
                retention_period=None,
                auto_promote=True,
                description="Active working memory"
            ),
            LevelSpec(
                name="warm",
                level_num=1,
                capacity=None,
                storage_type=StorageType.FILE_INDEXED,
                importance_threshold=0.4,
                retention_period="1d",
                auto_promote=True,
                description="Today's knowledge"
            ),
            LevelSpec(
                name="cool",
                level_num=2,
                capacity=None,
                storage_type=StorageType.FILE_INDEXED,
                importance_threshold=0.3,
                retention_period="7d",
                auto_promote=True,
                description="This week's knowledge"
            ),
            LevelSpec(
                name="cold",
                level_num=3,
                capacity=None,
                storage_type=StorageType.FILE_COLD,
                importance_threshold=0.1,
                retention_period="30d",
                auto_promote=True,
                compress=True,
                description="This month's knowledge"
            ),
            LevelSpec(
                name="archive",
                level_num=4,
                capacity=None,
                storage_type=StorageType.FILE_COLD,
                importance_threshold=0.0,
                retention_period="365d",
                auto_promote=True,
                compress=True,
                description="Long-term archive"
            )
        ]
    )


def create_regulatory_config() -> MemoryConfig:
    """
    6-level config for regulatory compliance with strict retention.
    
    - L0: Hot RAM (current session)
    - L1: Warm (7 days - reviewable)
    - L2: Cool (30 days - monthly audit)
    - L3: Cold (90 days - quarterly review)
    - L4: Archive (1 year - annual compliance)
    - L5: Permanent (forever - legal requirement)
    
    Use case: Healthcare, finance, legal, government
    """
    return MemoryConfig(
        name="regulatory",
        description="6-level memory for compliance (strict retention periods)",
        levels=[
            LevelSpec(
                name="hot",
                level_num=0,
                capacity=500,
                storage_type=StorageType.RAM,
                importance_threshold=0.6,
                retention_period=None,
                auto_promote=True,
                description="Current session"
            ),
            LevelSpec(
                name="warm",
                level_num=1,
                capacity=None,
                storage_type=StorageType.FILE_INDEXED,
                importance_threshold=0.5,
                retention_period="7d",
                auto_promote=False,  # Don't auto-promote for audit trail
                description="7-day reviewable period"
            ),
            LevelSpec(
                name="cool",
                level_num=2,
                capacity=None,
                storage_type=StorageType.FILE_INDEXED,
                importance_threshold=0.4,
                retention_period="30d",
                auto_promote=False,
                description="Monthly audit period"
            ),
            LevelSpec(
                name="cold",
                level_num=3,
                capacity=None,
                storage_type=StorageType.FILE_COLD,
                importance_threshold=0.2,
                retention_period="90d",
                auto_promote=False,
                compress=True,
                description="Quarterly review period"
            ),
            LevelSpec(
                name="archive",
                level_num=4,
                capacity=None,
                storage_type=StorageType.FILE_COLD,
                importance_threshold=0.0,
                retention_period="365d",
                auto_promote=False,
                compress=True,
                description="Annual compliance"
            ),
            LevelSpec(
                name="permanent",
                level_num=5,
                capacity=None,
                storage_type=StorageType.FILE_COLD,
                importance_threshold=0.0,
                retention_period=None,  # Forever
                auto_promote=False,
                compress=True,
                description="Legal permanent record"
            )
        ]
    )


def create_research_config() -> MemoryConfig:
    """
    4-level config for long-running research projects.
    
    - L0: Hot RAM (active research)
    - L1: Warm (recent findings)
    - L2: Cool (established knowledge)
    - L3: Cold (background/literature)
    
    Use case: Scientific research, long-term analysis
    """
    return MemoryConfig(
        name="research",
        description="4-level memory for research projects (weeks to months)",
        levels=[
            LevelSpec(
                name="hot",
                level_num=0,
                capacity=150,
                storage_type=StorageType.RAM,
                importance_threshold=0.5,
                retention_period=None,
                auto_promote=True,
                description="Active research context"
            ),
            LevelSpec(
                name="warm",
                level_num=1,
                capacity=None,
                storage_type=StorageType.FILE_INDEXED,
                importance_threshold=0.4,
                retention_period="7d",
                auto_promote=True,
                description="Recent findings"
            ),
            LevelSpec(
                name="cool",
                level_num=2,
                capacity=None,
                storage_type=StorageType.FILE_INDEXED,
                importance_threshold=0.2,
                retention_period="30d",
                auto_promote=True,
                description="Established knowledge"
            ),
            LevelSpec(
                name="cold",
                level_num=3,
                capacity=None,
                storage_type=StorageType.FILE_COLD,
                importance_threshold=0.0,
                retention_period="180d",
                auto_promote=True,
                compress=True,
                description="Background literature"
            )
        ]
    )


# ============================================================================
# PRESET REGISTRY
# ============================================================================

PRESETS: Dict[str, callable] = {
    "chatbot": create_chatbot_config,
    "assistant": create_assistant_config,
    "enterprise": create_enterprise_config,
    "regulatory": create_regulatory_config,
    "research": create_research_config,
}


def get_preset(preset_name: str) -> MemoryConfig:
    """
    Get a preset configuration by name.
    
    Args:
        preset_name: Name of the preset (chatbot, assistant, enterprise, regulatory, research)
        
    Returns:
        MemoryConfig for the preset
        
    Raises:
        ValueError: If preset name not found
    """
    preset_name = preset_name.lower()
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return PRESETS[preset_name]()


def list_presets() -> Dict[str, str]:
    """
    List all available presets with descriptions.
    
    Returns:
        Dict mapping preset names to descriptions
    """
    result = {}
    for name, factory in PRESETS.items():
        config = factory()
        result[name] = f"{config.description} ({config.num_levels} levels)"
    return result


def create_custom_config(
    num_levels: int = 3,
    base_preset: str = "assistant",
    hot_capacity: int = 100,
    retention_periods: Optional[List[Optional[str]]] = None,
    compress_cold: bool = True
) -> MemoryConfig:
    """
    Create a custom configuration based on parameters.
    
    Args:
        num_levels: Number of memory levels (2-7)
        base_preset: Base preset to derive from
        hot_capacity: Capacity of hot (L0) level
        retention_periods: List of retention periods for each level (None = forever)
        compress_cold: Compress colder levels
        
    Returns:
        Custom MemoryConfig
    """
    if num_levels < 2 or num_levels > 7:
        raise ValueError("Number of levels must be between 2 and 7")
    
    # Start with base preset
    base = get_preset(base_preset)
    
    # Generate level names
    if num_levels == 2:
        names = ["hot", "cold"]
    elif num_levels == 3:
        names = ["hot", "warm", "cold"]
    elif num_levels == 4:
        names = ["hot", "warm", "cool", "cold"]
    elif num_levels == 5:
        names = ["hot", "warm", "cool", "cold", "archive"]
    elif num_levels == 6:
        names = ["hot", "warm", "cool", "cold", "archive", "deep"]
    else:
        names = ["hot", "warm", "cool", "cold", "archive", "deep", "permanent"]
    
    # Default retention periods
    if retention_periods is None:
        default_retention = [None, "1d", "7d", "30d", "90d", "365d", None]
        retention_periods = default_retention[:num_levels]
    
    # Ensure retention_periods has right length
    while len(retention_periods) < num_levels:
        retention_periods.append("30d")
    retention_periods = retention_periods[:num_levels]
    
    # Create levels
    levels = []
    for i in range(num_levels):
        # Calculate importance threshold (decrease as we go colder)
        threshold = max(0.0, 0.6 - (i * 0.15))
        
        # Determine storage type
        if i == 0:
            storage = StorageType.RAM
        elif i < num_levels - 1:
            storage = StorageType.FILE_INDEXED
        else:
            storage = StorageType.FILE_COLD
        
        # Capacity (only hot level has limited capacity)
        capacity = hot_capacity if i == 0 else None
        
        # Compress colder levels
        compress = compress_cold and i >= num_levels - 2
        
        level = LevelSpec(
            name=names[i],
            level_num=i,
            capacity=capacity,
            storage_type=storage,
            importance_threshold=threshold,
            retention_period=retention_periods[i],
            auto_promote=True,
            compress=compress,
            description=f"Level {i}: {names[i]}"
        )
        levels.append(level)
    
    return MemoryConfig(
        name=f"custom_{num_levels}level",
        description=f"Custom {num_levels}-level configuration",
        levels=levels,
        auto_compact=True,
        compaction_threshold=0.8,
        decay_factor=0.95
    )