"""
Knowledge Graph - Core data structures for agent memory system.

This module implements a directed knowledge graph where nodes represent
knowledge units and edges represent relationships between them.
"""

import uuid
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from datetime import datetime


class NodeType(Enum):
    """Types of knowledge nodes."""
    FACT = "fact"
    EXPERIENCE = "experience"
    SKILL = "skill"
    RELATIONSHIP = "relationship"
    GOAL = "goal"
    DECISION = "decision"
    OBSERVATION = "observation"
    PLAN = "plan"
    ERROR = "error"
    SOLUTION = "solution"


class RelationType(Enum):
    """Types of relationships between nodes."""
    CAUSES = "causes"
    RELATES_TO = "relates_to"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    PART_OF = "part_of"
    LEADS_TO = "leads_to"
    DEPENDS_ON = "depends_on"
    SOLVES = "solves"
    PRECEDES = "precedes"
    SIMILAR_TO = "similar_to"


class StorageLevel(Enum):
    """Memory hierarchy levels."""
    L1_RAM = 1      # Hot memory - immediate access
    L2_WARM = 2     # Warm storage - indexed MD files
    L3_COLD = 3     # Cold storage - archived MD files


@dataclass
class Edge:
    """Represents a directed edge in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: str = "relates_to"
    weight: float = 1.0
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize edge to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Edge':
        """Deserialize edge from dictionary."""
        return cls(**data)


@dataclass
class KnowledgeNode:
    """
    Represents a single unit of knowledge in the graph.
    
    This is the fundamental building block of the memory system.
    Each node contains content, metadata, and references to connected nodes.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: str = NodeType.FACT.value
    content: str = ""
    summary: str = ""
    importance: float = 0.5
    last_accessed: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    storage_level: int = StorageLevel.L1_RAM.value
    
    # File pointer for compacted nodes
    file_path: Optional[str] = None
    file_offset: Optional[int] = None
    
    # Edges stored as list of edge IDs
    edge_ids: List[str] = field(default_factory=list)

    def access(self) -> None:
        """Mark node as accessed, updating access statistics."""
        self.last_accessed = datetime.utcnow().isoformat()
        self.access_count += 1

    def calculate_importance(self, decay_factor: float = 0.95) -> float:
        """
        Calculate current importance with time decay.
        
        Importance decays over time unless reinforced by access.
        """
        # Time since last access in hours
        last_access = datetime.fromisoformat(self.last_accessed)
        hours_since = (datetime.utcnow() - last_access).total_seconds() / 3600
        
        # Decay formula: importance * decay_factor^hours
        decayed = self.importance * (decay_factor ** (hours_since / 24))
        
        # Boost from access frequency
        frequency_boost = min(0.3, self.access_count * 0.01)
        
        return min(1.0, max(0.0, decayed + frequency_boost))

    def to_compact_dict(self) -> Dict:
        """
        Create compact representation for index storage.
        Used when node is stored in L2/L3 but index stays in parent level.
        """
        return {
            "id": self.id,
            "summary": self.summary,
            "tags": self.tags,
            "importance": self.importance,
            "node_type": self.node_type,
            "file_path": self.file_path,
            "file_offset": self.file_offset,
            "last_accessed": self.last_accessed,
            "created": self.created
        }

    def to_dict(self) -> Dict:
        """Serialize node to dictionary."""
        data = asdict(self)
        # Convert datetime objects to strings for JSON serialization
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeNode':
        """Deserialize node from dictionary."""
        return cls(**data)

    def to_markdown(self) -> str:
        """Convert node to markdown format for file storage."""
        edges_str = ", ".join([
            f"{eid}" for eid in self.edge_ids
        ])
        
        metadata_str = json.dumps(self.metadata, indent=2) if self.metadata else "{}"
        
        return f"""### Node: {self.id}
**Type:** {self.node_type}
**Created:** {self.created}
**Last Accessed:** {self.last_accessed}
**Importance:** {self.importance:.4f}
**Access Count:** {self.access_count}
**Tags:** {', '.join(self.tags) if self.tags else 'none'}
**Edges:** [{edges_str}]

#### Content
{self.content}

#### Summary
{self.summary}

#### Metadata
```json
{metadata_str}
```

---
[END NODE: {self.id}]
"""

    @classmethod
    def from_markdown(cls, md_content: str) -> 'KnowledgeNode':
        """Parse node from markdown format."""
        lines = md_content.strip().split('\n')
        node = cls()
        
        content_lines = []
        summary_lines = []
        metadata_lines = []
        current_section = None
        
        for line in lines:
            if line.startswith('### Node:'):
                node.id = line.split(':', 1)[1].strip()
            elif line.startswith('**Type:**'):
                node.node_type = line.split(':', 1)[1].strip()
            elif line.startswith('**Created:**'):
                node.created = line.split(':', 1)[1].strip()
            elif line.startswith('**Last Accessed:**'):
                node.last_accessed = line.split(':', 1)[1].strip()
            elif line.startswith('**Importance:**'):
                importance_str = line.split(':', 1)[1].strip()
                # Handle format like "** 0.6000"
                importance_str = importance_str.replace('**', '').strip()
                node.importance = float(importance_str)
            elif line.startswith('**Access Count:**'):
                count_str = line.split(':', 1)[1].strip()
                # Handle format like "** 1"
                count_str = count_str.replace('**', '').strip()
                node.access_count = int(count_str)
            elif line.startswith('**Tags:**'):
                tags_str = line.split(':', 1)[1].strip()
                node.tags = [t.strip() for t in tags_str.split(',') if t.strip() and t.strip() != 'none']
            elif line.startswith('**Edges:**'):
                edges_str = line.split(':', 1)[1].strip()
                # Parse [edge1, edge2, ...]
                edges_str = edges_str.strip('[]')
                node.edge_ids = [e.strip() for e in edges_str.split(',') if e.strip()]
            elif line.startswith('#### Content'):
                current_section = 'content'
            elif line.startswith('#### Summary'):
                current_section = 'summary'
            elif line.startswith('#### Metadata'):
                current_section = 'metadata'
            elif line.startswith('```json'):
                continue
            elif line.startswith('```'):
                current_section = None
            elif line.startswith('[END NODE:'):
                break
            else:
                if current_section == 'content':
                    content_lines.append(line)
                elif current_section == 'summary':
                    summary_lines.append(line)
                elif current_section == 'metadata':
                    metadata_lines.append(line)
        
        node.content = '\n'.join(content_lines).strip()
        node.summary = '\n'.join(summary_lines).strip()
        
        if metadata_lines:
            try:
                node.metadata = json.loads('\n'.join(metadata_lines))
            except json.JSONDecodeError:
                node.metadata = {}
        
        return node


class KnowledgeGraph:
    """
    Manages the complete knowledge graph structure.
    
    Provides operations for adding, removing, and querying nodes and edges.
    """
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, Edge] = {}
        
        # Adjacency lists for fast traversal
        self.outgoing: Dict[str, Set[str]] = {}  # node_id -> set of edge_ids
        self.incoming: Dict[str, Set[str]] = {}  # node_id -> set of edge_ids

    def add_node(self, node: KnowledgeNode) -> str:
        """Add a node to the graph. Returns node ID."""
        self.nodes[node.id] = node
        if node.id not in self.outgoing:
            self.outgoing[node.id] = set()
        if node.id not in self.incoming:
            self.incoming[node.id] = set()
        return node.id

    def remove_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Remove a node and its edges from the graph.
        Returns the removed node or None if not found.
        """
        if node_id not in self.nodes:
            return None
        
        node = self.nodes.pop(node_id)
        
        # Remove all edges connected to this node
        edges_to_remove = set()
        if node_id in self.outgoing:
            edges_to_remove.update(self.outgoing.pop(node_id))
        if node_id in self.incoming:
            edges_to_remove.update(self.incoming.pop(node_id))
        
        for edge_id in edges_to_remove:
            self.edges.pop(edge_id, None)
        
        # Clean up adjacency lists
        for adj_set in self.outgoing.values():
            adj_set -= edges_to_remove
        for adj_set in self.incoming.values():
            adj_set -= edges_to_remove
        
        return node

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def add_edge(self, edge: Edge) -> str:
        """Add an edge to the graph. Returns edge ID."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(f"Source or target node not found: {edge.source_id} -> {edge.target_id}")
        
        self.edges[edge.id] = edge
        
        if edge.source_id not in self.outgoing:
            self.outgoing[edge.source_id] = set()
        self.outgoing[edge.source_id].add(edge.id)
        
        if edge.target_id not in self.incoming:
            self.incoming[edge.target_id] = set()
        self.incoming[edge.target_id].add(edge.id)
        
        # Add edge ID to node's edge list
        self.nodes[edge.source_id].edge_ids.append(edge.id)
        self.nodes[edge.target_id].edge_ids.append(edge.id)
        
        return edge.id

    def remove_edge(self, edge_id: str) -> Optional[Edge]:
        """Remove an edge from the graph."""
        if edge_id not in self.edges:
            return None
        
        edge = self.edges.pop(edge_id)
        
        if edge.source_id in self.outgoing:
            self.outgoing[edge.source_id].discard(edge_id)
        if edge.target_id in self.incoming:
            self.incoming[edge.target_id].discard(edge_id)
        
        # Remove from node edge lists
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].edge_ids = [
                eid for eid in self.nodes[edge.source_id].edge_ids if eid != edge_id
            ]
        if edge.target_id in self.nodes:
            self.nodes[edge.target_id].edge_ids = [
                eid for eid in self.nodes[edge.target_id].edge_ids if eid != edge_id
            ]
        
        return edge

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """
        Get neighboring node IDs.
        
        Args:
            node_id: The node to find neighbors for
            direction: "outgoing", "incoming", or "both"
        """
        neighbors = set()
        
        if direction in ("outgoing", "both") and node_id in self.outgoing:
            for edge_id in self.outgoing[node_id]:
                edge = self.edges.get(edge_id)
                if edge:
                    neighbors.add(edge.target_id)
        
        if direction in ("incoming", "both") and node_id in self.incoming:
            for edge_id in self.incoming[node_id]:
                edge = self.edges.get(edge_id)
                if edge:
                    neighbors.add(edge.source_id)
        
        return list(neighbors)

    def get_edges_between(self, source_id: str, target_id: str) -> List[Edge]:
        """Get all edges between two nodes."""
        result = []
        if source_id in self.outgoing:
            for edge_id in self.outgoing[source_id]:
                edge = self.edges.get(edge_id)
                if edge and edge.target_id == target_id:
                    result.append(edge)
        return result

    def find_nodes_by_tag(self, tag: str) -> List[KnowledgeNode]:
        """Find all nodes with a specific tag."""
        return [node for node in self.nodes.values() if tag in node.tags]

    def find_nodes_by_type(self, node_type: str) -> List[KnowledgeNode]:
        """Find all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def search_content(self, query: str) -> List[KnowledgeNode]:
        """
        Simple text search across node content and summaries.
        Returns nodes sorted by relevance (importance).
        """
        query_lower = query.lower()
        matches = []
        
        for node in self.nodes.values():
            if (query_lower in node.content.lower() or 
                query_lower in node.summary.lower() or
                any(query_lower in tag.lower() for tag in node.tags)):
                matches.append(node)
        
        # Sort by importance
        matches.sort(key=lambda n: n.calculate_importance(), reverse=True)
        return matches

    def get_subgraph(self, node_ids: Set[str]) -> 'KnowledgeGraph':
        """Extract a subgraph containing only specified nodes."""
        subgraph = KnowledgeGraph()
        
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
        
        for edge in self.edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                subgraph.add_edge(edge)
        
        return subgraph

    def to_dict(self) -> Dict:
        """Serialize graph to dictionary."""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": {eid: edge.to_dict() for eid, edge in self.edges.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeGraph':
        """Deserialize graph from dictionary."""
        graph = cls()
        
        for node_data in data.get("nodes", {}).values():
            node = KnowledgeNode.from_dict(node_data)
            graph.add_node(node)
        
        for edge_data in data.get("edges", {}).values():
            edge = Edge.from_dict(edge_data)
            graph.add_edge(edge)
        
        return graph

    @property
    def node_count(self) -> int:
        """Get number of nodes in graph."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Get number of edges in graph."""
        return len(self.edges)