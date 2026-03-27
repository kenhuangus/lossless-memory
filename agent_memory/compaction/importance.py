"""
Importance Scorer - Calculates and updates node importance scores.

This module implements scoring algorithms to determine which nodes
should remain in hot memory vs being compacted to cold storage.
"""

import math
from datetime import datetime, timedelta
from typing import List, Optional

from ..knowledge_graph import KnowledgeNode, NodeType


class ImportanceScorer:
    """
    Calculates importance scores for knowledge nodes.
    
    Factors considered:
    1. Recency of access
    2. Frequency of access
    3. Node type (decisions and errors are more important)
    4. Number of connections (well-connected nodes are more important)
    5. User-specified importance
    6. Time decay
    """
    
    # Type weights - some node types are inherently more important
    TYPE_WEIGHTS = {
        NodeType.DECISION.value: 1.3,
        NodeType.ERROR.value: 1.2,
        NodeType.SOLUTION.value: 1.2,
        NodeType.GOAL.value: 1.15,
        NodeType.PLAN.value: 1.1,
        NodeType.SKILL.value: 1.1,
        NodeType.EXPERIENCE.value: 1.0,
        NodeType.FACT.value: 0.9,
        NodeType.OBSERVATION.value: 0.85,
        NodeType.RELATIONSHIP.value: 1.0
    }
    
    def __init__(
        self,
        decay_factor: float = 0.95,
        decay_period_hours: float = 24.0,
        recency_weight: float = 0.4,
        frequency_weight: float = 0.3,
        connectivity_weight: float = 0.2,
        type_weight: float = 0.1
    ):
        """
        Initialize importance scorer.
        
        Args:
            decay_factor: Base decay rate per period
            decay_period_hours: Hours per decay period
            recency_weight: Weight for recency score component
            frequency_weight: Weight for frequency score component
            connectivity_weight: Weight for connectivity score component
            type_weight: Weight for node type score component
        """
        self.decay_factor = decay_factor
        self.decay_period_hours = decay_period_hours
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.connectivity_weight = connectivity_weight
        self.type_weight = type_weight
        
        # Normalize weights to sum to 1.0
        total = (recency_weight + frequency_weight + 
                 connectivity_weight + type_weight)
        self.recency_weight /= total
        self.frequency_weight /= total
        self.connectivity_weight /= total
        self.type_weight /= total

    def calculate_recency_score(self, node: KnowledgeNode) -> float:
        """
        Calculate score based on how recently the node was accessed.
        
        Returns value between 0 and 1.
        """
        try:
            last_access = datetime.fromisoformat(node.last_accessed)
            hours_since = (datetime.utcnow() - last_access).total_seconds() / 3600
            
            # Exponential decay
            periods = hours_since / self.decay_period_hours
            score = self.decay_factor ** periods
            
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.5  # Default score for nodes with invalid dates

    def calculate_frequency_score(self, node: KnowledgeNode) -> float:
        """
        Calculate score based on how frequently the node is accessed.
        
        Uses logarithmic scaling to prevent frequently accessed nodes
        from dominating forever.
        
        Returns value between 0 and 1.
        """
        if node.access_count == 0:
            return 0.1
        
        # Log scaling: log(access_count + 1) / log(100)
        # This means 99 accesses = score of 1.0
        score = math.log(node.access_count + 1) / math.log(100)
        
        return max(0.0, min(1.0, score))

    def calculate_connectivity_score(self, node: KnowledgeNode, edge_count: int) -> float:
        """
        Calculate score based on how well-connected the node is.
        
        Well-connected nodes are more important as they serve as
        hubs in the knowledge graph.
        
        Returns value between 0 and 1.
        """
        if edge_count == 0:
            return 0.1
        
        # Log scaling: log(edges + 1) / log(20)
        # This means 19 connections = score of 1.0
        score = math.log(edge_count + 1) / math.log(20)
        
        return max(0.0, min(1.0, score))

    def calculate_type_score(self, node: KnowledgeNode) -> float:
        """
        Calculate score based on node type.
        
        Some types (decisions, errors, goals) are inherently more important.
        
        Returns value between 0 and 1.
        """
        return self.TYPE_WEIGHTS.get(node.node_type, 1.0)

    def calculate_importance(
        self, 
        node: KnowledgeNode, 
        edge_count: int = 0
    ) -> float:
        """
        Calculate overall importance score for a node.
        
        Args:
            node: The knowledge node to score
            edge_count: Number of edges connected to this node
            
        Returns:
            Importance score between 0 and 1
        """
        recency = self.calculate_recency_score(node)
        frequency = self.calculate_frequency_score(node)
        connectivity = self.calculate_connectivity_score(node, edge_count)
        type_score = self.calculate_type_score(node)
        
        # Weighted combination
        importance = (
            self.recency_weight * recency +
            self.frequency_weight * frequency +
            self.connectivity_weight * connectivity +
            self.type_weight * type_score
        )
        
        # Apply node's own importance multiplier (user-specified)
        importance *= node.importance
        
        return max(0.0, min(1.0, importance))

    def calculate_batch_importance(
        self,
        nodes: List[KnowledgeNode],
        edge_counts: dict = None
    ) -> dict:
        """
        Calculate importance scores for multiple nodes.
        
        Args:
            nodes: List of nodes to score
            edge_counts: Dict mapping node_id to edge count
            
        Returns:
            Dict mapping node_id to importance score
        """
        if edge_counts is None:
            edge_counts = {}
        
        scores = {}
        for node in nodes:
            edge_count = edge_counts.get(node.id, len(node.edge_ids))
            scores[node.id] = self.calculate_importance(node, edge_count)
        
        return scores

    def update_node_importance(
        self,
        node: KnowledgeNode,
        edge_count: int = 0
    ) -> float:
        """
        Update and return the importance score for a node.
        
        This modifies the node's importance field.
        """
        new_importance = self.calculate_importance(node, edge_count)
        node.importance = new_importance
        return new_importance

    def rank_nodes(
        self,
        nodes: List[KnowledgeNode],
        edge_counts: dict = None,
        ascending: bool = False
    ) -> List[tuple]:
        """
        Rank nodes by importance.
        
        Args:
            nodes: List of nodes to rank
            edge_counts: Dict mapping node_id to edge count
            ascending: If True, least important first
            
        Returns:
            List of (node, importance_score) tuples sorted by importance
        """
        scores = self.calculate_batch_importance(nodes, edge_counts)
        
        ranked = [(node, scores.get(node.id, 0.0)) for node in nodes]
        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        
        return ranked

    def get_low_importance_nodes(
        self,
        nodes: List[KnowledgeNode],
        threshold: float = 0.3,
        edge_counts: dict = None
    ) -> List[KnowledgeNode]:
        """
        Get nodes with importance below threshold.
        
        These are candidates for compaction to lower storage levels.
        """
        scores = self.calculate_batch_importance(nodes, edge_counts)
        
        return [
            node for node in nodes
            if scores.get(node.id, 0.0) < threshold
        ]