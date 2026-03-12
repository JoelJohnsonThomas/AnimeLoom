"""
Workflow Graph - Shot dependency management for the Director Agent.
Tracks which shots depend on others (e.g., same character, same scene)
and determines optimal execution order.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


class WorkflowGraph:
    """
    Directed acyclic graph for shot dependencies.

    Nodes are shot indices. An edge (A -> B) means shot B depends on shot A
    (e.g., same character must be generated first in shot A so LoRA is ready).
    """

    def __init__(self):
        self.adjacency: Dict[int, List[int]] = defaultdict(list)
        self.in_degree: Dict[int, int] = defaultdict(int)
        self.shot_metadata: Dict[int, Dict] = {}

    def add_shot(self, shot_index: int, metadata: Dict = None):
        """Register a shot in the graph."""
        if shot_index not in self.in_degree:
            self.in_degree[shot_index] = 0
        self.shot_metadata[shot_index] = metadata or {}

    def add_dependency(self, prerequisite: int, dependent: int):
        """Declare that *dependent* cannot run until *prerequisite* completes."""
        self.adjacency[prerequisite].append(dependent)
        self.in_degree[dependent] = self.in_degree.get(dependent, 0) + 1

    def topological_order(self) -> List[int]:
        """Return shots in a valid execution order (Kahn's algorithm)."""
        queue = deque(
            node for node, deg in self.in_degree.items() if deg == 0
        )
        order: List[int] = []

        temp_in = dict(self.in_degree)
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in self.adjacency.get(node, []):
                temp_in[neighbour] -= 1
                if temp_in[neighbour] == 0:
                    queue.append(neighbour)

        if len(order) != len(self.in_degree):
            raise ValueError("Cycle detected in shot dependency graph")

        return order

    def get_parallelisable_groups(self) -> List[List[int]]:
        """
        Return groups of shots that can be processed in parallel.
        Each group contains shots whose dependencies are already satisfied
        by all previous groups.
        """
        groups: List[List[int]] = []
        temp_in = dict(self.in_degree)
        remaining: Set[int] = set(self.in_degree.keys())

        while remaining:
            # Shots with zero in-degree in the remaining set
            ready = [n for n in remaining if temp_in[n] == 0]
            if not ready:
                raise ValueError("Cycle detected in shot dependency graph")
            groups.append(sorted(ready))
            for node in ready:
                remaining.discard(node)
                for neighbour in self.adjacency.get(node, []):
                    temp_in[neighbour] -= 1

        return groups

    @staticmethod
    def build_from_shots(shots: List[Dict]) -> "WorkflowGraph":
        """
        Automatically build a dependency graph from parsed shots.

        Rule: if a character appears in shot *i* and later in shot *j* (j > i),
        shot *j* depends on the **first** shot that introduces the character
        (so the LoRA is trained / loaded before re-use).
        """
        graph = WorkflowGraph()
        first_appearance: Dict[str, int] = {}

        for idx, shot in enumerate(shots):
            graph.add_shot(idx, metadata=shot)
            for char_name in shot.get("characters", []):
                if char_name in first_appearance:
                    # Only add dependency to the *first* appearance
                    if first_appearance[char_name] != idx:
                        graph.add_dependency(first_appearance[char_name], idx)
                else:
                    first_appearance[char_name] = idx

        return graph

    def __repr__(self) -> str:
        edges = []
        for src, dsts in self.adjacency.items():
            for dst in dsts:
                edges.append(f"{src}->{dst}")
        return f"WorkflowGraph(shots={list(self.in_degree.keys())}, edges=[{', '.join(edges)}])"
