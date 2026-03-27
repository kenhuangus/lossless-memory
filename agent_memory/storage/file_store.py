"""
File Store (L2/L3) - Persistent storage using markdown files.

This module handles storing knowledge nodes in markdown files
organized by topic/session, with efficient retrieval via byte offsets.
"""

import os
import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from ..knowledge_graph import KnowledgeNode, StorageLevel


class FileStore:
    """
    Level 2/3 storage using markdown files.
    
    Features:
    - Nodes stored in human-readable markdown format
    - Organized by shard (topic/session based)
    - Fast retrieval via byte offsets
    - Supports both L2 (warm) and L3 (cold) storage
    """
    
    # Separator between nodes in markdown files
    NODE_SEPARATOR = "---\n"
    
    def __init__(self, base_path: str, level: int = 2, max_shard_size: int = 50):
        """
        Initialize file store.
        
        Args:
            base_path: Base directory for storage files
            level: Storage level (2 for warm, 3 for cold)
            max_shard_size: Maximum nodes per shard file
        """
        self.base_path = Path(base_path)
        self.level = level
        self.max_shard_size = max_shard_size
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Track shard contents: shard_id -> list of (node_id, byte_offset)
        self.shard_index: Dict[str, List[Tuple[str, int]]] = {}
        
        # Track node to shard mapping: node_id -> (shard_id, byte_offset)
        self.node_locations: Dict[str, Tuple[str, int]] = {}
        
        # Initialize by scanning existing files
        self._scan_existing_shards()

    def _scan_existing_shards(self) -> None:
        """Scan existing shard files and build index."""
        for shard_file in self.base_path.glob("*.md"):
            shard_id = shard_file.stem
            self._build_shard_index(shard_id, shard_file)

    def _build_shard_index(self, shard_id: str, shard_path: Path) -> None:
        """Build index for a single shard file."""
        if not shard_path.exists():
            return
        
        content = shard_path.read_text(encoding='utf-8')
        
        # Find all node markers
        node_pattern = r'### Node: ([a-f0-9-]+)'
        positions = []
        
        for match in re.finditer(node_pattern, content):
            node_id = match.group(1)
            offset = match.start()
            positions.append((node_id, offset))
            self.node_locations[node_id] = (shard_id, offset)
        
        self.shard_index[shard_id] = positions

    def _get_shard_path(self, shard_id: str) -> Path:
        """Get file path for a shard."""
        return self.base_path / f"{shard_id}.md"

    def _get_shard_header(self, shard_id: str, node_count: int) -> str:
        """Generate header for a shard file."""
        return f"""# Memory Shard: {shard_id}
## Level: L{self.level}
## Created: {datetime.utcnow().isoformat()}
## Node Count: {node_count}

"""

    def _find_available_shard(self, tags: List[str] = None) -> str:
        """
        Find or create an available shard for storing nodes.
        
        Uses tags to group related nodes when possible.
        """
        # Sanitize tags first
        if tags:
            sanitized_tags = []
            for tag in tags:
                if tag and not tag.isspace():
                    # Remove invalid file path characters
                    clean_tag = tag.lower().replace(" ", "_")
                    invalid_chars = '<>:"/\\|?*'
                    for char in invalid_chars:
                        clean_tag = clean_tag.replace(char, '')
                    if clean_tag and not clean_tag.isspace():
                        sanitized_tags.append(clean_tag)
            tags = sanitized_tags if sanitized_tags else None
        
        # Try to find existing shard with matching tags
        if tags:
            for shard_id, positions in self.shard_index.items():
                if len(positions) < self.max_shard_size:
                    # Check if shard name relates to tags
                    if any(tag in shard_id.lower() for tag in tags):
                        return shard_id
        
        # Find any shard with capacity
        for shard_id, positions in self.shard_index.items():
            if len(positions) < self.max_shard_size:
                return shard_id
        
        # Create new shard
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        # Use first sanitized tag or default to "general"
        tag_hint = tags[0] if tags else "general"
        shard_id = f"L{self.level}_{tag_hint}_{timestamp}"
        
        # Initialize empty shard
        self.shard_index[shard_id] = []
        
        return shard_id

    def store_node(self, node: KnowledgeNode) -> Tuple[str, int]:
        """
        Store a node to file storage.
        
        Returns:
            Tuple of (shard_id, byte_offset)
        """
        # Set storage level
        node.storage_level = self.level
        
        # Find appropriate shard
        shard_id = self._find_available_shard(node.tags)
        shard_path = self._get_shard_path(shard_id)
        
        # Convert node to markdown
        node_md = node.to_markdown()
        
        # Check if shard exists
        if shard_path.exists():
            # Append to existing shard
            with open(shard_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            # Update node count in header
            current_count = len(self.shard_index.get(shard_id, []))
            new_count = current_count + 1
            
            # Replace node count in header
            existing_content = re.sub(
                r'## Node Count: \d+',
                f'## Node Count: {new_count}',
                existing_content
            )
            
            # Append new node
            byte_offset = len(existing_content.encode('utf-8'))
            full_content = existing_content + self.NODE_SEPARATOR + node_md
            
            with open(shard_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
        else:
            # Create new shard with header
            header = self._get_shard_header(shard_id, 1)
            full_content = header + node_md
            byte_offset = len(header.encode('utf-8'))
            
            with open(shard_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
        
        # Update indices
        if shard_id not in self.shard_index:
            self.shard_index[shard_id] = []
        self.shard_index[shard_id].append((node.id, byte_offset))
        self.node_locations[node.id] = (shard_id, byte_offset)
        
        # Update node's file pointer
        node.file_path = str(shard_path)
        node.file_offset = byte_offset
        
        return shard_id, byte_offset

    def load_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """
        Load a node from file storage by ID.
        
        Returns None if node not found.
        """
        if node_id not in self.node_locations:
            return None
        
        shard_id, byte_offset = self.node_locations[node_id]
        shard_path = self._get_shard_path(shard_id)
        
        if not shard_path.exists():
            return None
        
        try:
            with open(shard_path, 'r', encoding='utf-8') as f:
                f.seek(byte_offset)
                
                # Read until next node separator or end marker
                content = []
                for line in f:
                    if line.startswith('[END NODE:'):
                        content.append(line)
                        break
                    content.append(line)
                
                node_md = ''.join(content)
                return KnowledgeNode.from_markdown(node_md)
        except Exception as e:
            print(f"Error loading node {node_id}: {e}")
            return None

    def load_node_by_pointer(self, file_path: str, byte_offset: int) -> Optional[KnowledgeNode]:
        """
        Load a node using file path and byte offset.
        
        Useful when node location is known from index.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.seek(byte_offset)
                
                content = []
                for line in f:
                    if line.startswith('[END NODE:'):
                        content.append(line)
                        break
                    content.append(line)
                
                node_md = ''.join(content)
                return KnowledgeNode.from_markdown(node_md)
        except Exception as e:
            print(f"Error loading node from {file_path}:{byte_offset}: {e}")
            return None

    def remove_node(self, node_id: str) -> bool:
        """
        Mark a node as removed (soft delete).
        
        Note: Actual file content is not modified to maintain lossless guarantee.
        Returns True if node was found and marked.
        """
        if node_id not in self.node_locations:
            return False
        
        shard_id, _ = self.node_locations[node_id]
        
        # Remove from indices (but file content remains)
        if shard_id in self.shard_index:
            self.shard_index[shard_id] = [
                (nid, offset) for nid, offset in self.shard_index[shard_id]
                if nid != node_id
            ]
        
        del self.node_locations[node_id]
        
        return True

    def list_shards(self) -> List[str]:
        """List all shard IDs."""
        return list(self.shard_index.keys())

    def get_shard_info(self, shard_id: str) -> Dict:
        """Get information about a shard."""
        positions = self.shard_index.get(shard_id, [])
        shard_path = self._get_shard_path(shard_id)
        
        return {
            "shard_id": shard_id,
            "node_count": len(positions),
            "file_exists": shard_path.exists(),
            "file_size": shard_path.stat().st_size if shard_path.exists() else 0,
            "node_ids": [nid for nid, _ in positions]
        }

    def search_shard_content(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search across shard contents.
        
        Note: This is slower than index search as it reads files.
        """
        query_lower = query.lower()
        results = []
        
        for shard_id, positions in self.shard_index.items():
            shard_path = self._get_shard_path(shard_id)
            if not shard_path.exists():
                continue
            
            try:
                with open(shard_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple text search in content
                if query_lower in content.lower():
                    # Find which nodes match
                    for node_id, offset in positions:
                        # Extract node content (approximate)
                        node_start = offset
                        node_end = content.find('[END NODE:', node_start)
                        if node_end == -1:
                            node_end = len(content)
                        
                        node_content = content[node_start:node_end]
                        if query_lower in node_content.lower():
                            results.append({
                                "node_id": node_id,
                                "shard_id": shard_id,
                                "offset": offset
                            })
                            
                            if len(results) >= limit:
                                return results
            except Exception as e:
                print(f"Error searching shard {shard_id}: {e}")
                continue
        
        return results

    def get_all_node_locations(self) -> Dict[str, Tuple[str, int]]:
        """Get all node locations (node_id -> (shard_id, offset))."""
        return self.node_locations.copy()

    def compact_shard(self, shard_id: str) -> bool:
        """
        Compact a shard by removing gaps and updating offsets.
        
        This is a safe operation as it doesn't remove any content,
        just reorganizes it.
        """
        shard_path = self._get_shard_path(shard_id)
        if not shard_path.exists():
            return False
        
        try:
            # Read all content
            with open(shard_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse all nodes
            nodes = []
            node_pattern = r'### Node: [a-f0-9-]+.*?\[END NODE: [a-f0-9-]+\]'
            
            for match in re.finditer(node_pattern, content, re.DOTALL):
                node_content = match.group(0)
                try:
                    node = KnowledgeNode.from_markdown(node_content)
                    nodes.append(node)
                except Exception as e:
                    print(f"Error parsing node: {e}")
            
            if not nodes:
                return False
            
            # Rebuild file with updated header
            header = self._get_shard_header(shard_id, len(nodes))
            new_content = header
            
            # Update offsets in index
            new_positions = []
            for node in nodes:
                offset = len(new_content.encode('utf-8'))
                new_content += node.to_markdown() + self.NODE_SEPARATOR
                new_positions.append((node.id, offset))
            
            # Write compacted file
            with open(shard_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Update indices
            self.shard_index[shard_id] = new_positions
            for node_id, offset in new_positions:
                self.node_locations[node_id] = (shard_id, offset)
            
            return True
            
        except Exception as e:
            print(f"Error compacting shard {shard_id}: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get statistics about the file store."""
        total_nodes = sum(len(positions) for positions in self.shard_index.values())
        total_size = sum(
            self._get_shard_path(sid).stat().st_size 
            for sid in self.shard_index 
            if self._get_shard_path(sid).exists()
        )
        
        return {
            "level": self.level,
            "shard_count": len(self.shard_index),
            "total_nodes": total_nodes,
            "total_size_bytes": total_size,
            "avg_nodes_per_shard": total_nodes / len(self.shard_index) if self.shard_index else 0
        }