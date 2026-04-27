"""
DAG System for Model Evolution Graph
NOT a ledger - a model evolution DAG where:
- Nodes = validated model updates
- Edges A → B mean "B was trained from model state A"
- Supports concurrent sibling updates
- Sharded sub-graphs per shard
"""

import hashlib
import time
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


class UpdateStatus(Enum):
    """Status of a DAG node (model update)"""
    PENDING = "pending"          # Received, awaiting verification
    VERIFIED = "verified"        # ZKP verified, ready for aggregation
    AGGREGATED = "aggregated"    # Used in shard aggregation
    FINALIZED = "finalized"      # Permanently added to shard DAG
    REJECTED = "rejected"        # Failed verification or aggregation


@dataclass
class DAGNode:
    """
    Node in model evolution DAG
    Represents a validated model update
    """
    node_id: str  # Unique identifier
    client_id: str  # Client who produced update
    round_num: int  # Training round
    shard_id: str  # Which shard this belongs to
    
    # Model state
    model_hash: str  # Hash of model snapshot
    gradient_hash: str  # Hash of gradient update
    
    # DAG structure
    parent_id: str  # Previous model state this was trained from
    timestamp: float = field(default_factory=time.time)
    
    # Verification
    status: UpdateStatus = UpdateStatus.PENDING
    zkp_verified: bool = False
    commitment_verified: bool = False
    
    # Auxiliary
    reputation_score: float = 0.8  # Client reputation when created
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_hash(self) -> str:
        """Get cryptographic hash of this node"""
        content = f"{self.client_id}:{self.round_num}:{self.model_hash}:{self.gradient_hash}:{self.parent_id}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class DAGEdge:
    """Edge in model evolution DAG"""
    from_node_id: str  # Parent node
    to_node_id: str    # Child node
    edge_type: str = "parent_child"  # Type of relationship
    timestamp: float = field(default_factory=time.time)


class DAGGraph:
    """
    Local DAG for model evolution
    Each shard maintains its own DAG subgraph
    """
    
    def __init__(self, shard_id: str, initial_model_hash: str):
        """
        Initialize DAG for a shard.
        
        Args:
            shard_id: Identifier of shard
            initial_model_hash: Hash of initial model state
        """
        self.shard_id = shard_id
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: Dict[Tuple[str, str], DAGEdge] = {}
        self.root_hash = initial_model_hash
        self.max_concurrent_siblings = 5
        
        # Initialize root node
        root_node = DAGNode(
            node_id="root",
            client_id="system",
            round_num=0,
            shard_id=shard_id,
            model_hash=initial_model_hash,
            gradient_hash="initial",
            parent_id=None,
            status=UpdateStatus.FINALIZED,
            zkp_verified=False,
        )
        self.nodes["root"] = root_node
        
    def add_update(self, client_id: str, round_num: int,
                  model_hash: str, gradient_hash: str,
                  parent_id: str, metadata: Dict = None) -> str:
        """
        Add a model update to the DAG (pending verification).
        
        Args:
            client_id: Client who produced the update
            round_num: Training round number
            model_hash: Hash of resulting model
            gradient_hash: Hash of gradient
            parent_id: Parent node ID
            metadata: Additional metadata
            
        Returns:
            Node ID of the new update
        """
        import uuid
        node_id = str(uuid.uuid4())
        
        node = DAGNode(
            node_id=node_id,
            client_id=client_id,
            round_num=round_num,
            shard_id=self.shard_id,
            model_hash=model_hash,
            gradient_hash=gradient_hash,
            parent_id=parent_id,
            status=UpdateStatus.PENDING,
            metadata=metadata or {},
        )
        
        self.nodes[node_id] = node
        
        # Add edge from parent
        if parent_id and parent_id in self.nodes:
            edge_key = (parent_id, node_id)
            self.edges[edge_key] = DAGEdge(parent_id, node_id)
        
        return node_id
    
    def verify_update(self, node_id: str, 
                     zkp_verified: bool = True,
                     commitment_verified: bool = True) -> bool:
        """
        Mark update as verified and promote to VERIFIED status.
        
        Args:
            node_id: Node to verify
            zkp_verified: Whether ZKP was verified
            commitment_verified: Whether commitment was verified
            
        Returns:
            True if successfully verified
        """
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        node.zkp_verified = zkp_verified
        node.commitment_verified = commitment_verified
        node.status = UpdateStatus.VERIFIED
        return True
    
    def finalize_update(self, node_id: str) -> bool:
        """
        Finalize update - permanently add to DAG.
        Updates must be VERIFIED before finalization.
        
        Args:
            node_id: Node to finalize
            
        Returns:
            True if successfully finalized
        """
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        if node.status != UpdateStatus.VERIFIED:
            return False
        
        # Check sibling constraint
        if node.parent_id:
            siblings = self.get_siblings(node.parent_id)
            finalized_siblings = [
                n for n in siblings 
                if self.nodes[n].status == UpdateStatus.FINALIZED
            ]
            if len(finalized_siblings) >= self.max_concurrent_siblings:
                return False
        
        node.status = UpdateStatus.FINALIZED
        return True
    
    def reject_update(self, node_id: str, reason: str = "") -> bool:
        """
        Reject update and mark as failed verification.
        
        Args:
            node_id: Node to reject
            reason: Rejection reason
            
        Returns:
            True if successfully rejected
        """
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        node.status = UpdateStatus.REJECTED
        node.metadata["rejection_reason"] = reason
        return True
    
    def get_siblings(self, parent_id: str) -> List[str]:
        """Get all sibling updates (children of same parent)"""
        if parent_id not in self.nodes:
            return []
        
        siblings = []
        for (from_id, to_id), edge in self.edges.items():
            if from_id == parent_id:
                siblings.append(to_id)
        return siblings
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        """Get path from node to root"""
        path = []
        current = node_id
        visited = set()
        
        while current and current not in visited:
            visited.add(current)
            path.append(current)
            if current in self.nodes:
                current = self.nodes[current].parent_id
            else:
                break
        
        return path
    
    def get_finalized_chain(self) -> List[str]:
        """Get chain of finalized updates (main trunk)"""
        chain = ["root"]
        current_level = ["root"]
        
        while current_level:
            next_level = []
            for parent_id in current_level:
                children = self.get_siblings(parent_id)
                finalized = [c for c in children if self.nodes[c].status == UpdateStatus.FINALIZED]
                for child in finalized:
                    chain.append(child)
                    next_level.append(child)
            current_level = next_level
        
        return chain
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DAG statistics"""
        statuses = {}
        for node in self.nodes.values():
            s = node.status.value
            statuses[s] = statuses.get(s, 0) + 1
        
        finalized = [n for n in self.nodes.values() if n.status == UpdateStatus.FINALIZED]
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "status_breakdown": statuses,
            "finalized_chain_length": len(self.get_finalized_chain()),
            "num_clients": len(set(n.client_id for n in self.nodes.values())),
        }


class ShardedDAG:
    """
    Multi-shard DAG system
    Each shard has its own DAG subgraph
    Global aggregator can see all shards
    """
    
    def __init__(self, num_shards: int, initial_model_hash: str):
        """
        Initialize sharded DAG.
        
        Args:
            num_shards: Number of shards
            initial_model_hash: Hash of initial global model
        """
        self.num_shards = num_shards
        self.shard_dags: Dict[str, DAGGraph] = {}
        self.global_model_history: List[str] = [initial_model_hash]
        
        # Create DAG for each shard
        for i in range(num_shards):
            shard_id = f"shard_{i}"
            self.shard_dags[shard_id] = DAGGraph(shard_id, initial_model_hash)
    
    def add_shard_update(self, shard_id: str,
                        client_id: str, round_num: int,
                        model_hash: str, gradient_hash: str,
                        parent_id: str) -> str:
        """
        Add update to a shard's DAG.
        
        Args:
            shard_id: Target shard
            client_id: Client who produced update
            round_num: Training round
            model_hash: Model hash
            gradient_hash: Gradient hash
            parent_id: Parent node
            
        Returns:
            Node ID
        """
        if shard_id not in self.shard_dags:
            raise ValueError(f"Unknown shard: {shard_id}")
        
        return self.shard_dags[shard_id].add_update(
            client_id, round_num, model_hash, gradient_hash, parent_id
        )
    
    def record_global_update(self, model_hash: str) -> None:
        """Record global model update"""
        self.global_model_history.append(model_hash)
    
    def get_all_finalized_updates(self) -> Dict[str, List[DAGNode]]:
        """Get all finalized updates per shard"""
        result = {}
        for shard_id, dag in self.shard_dags.items():
            finalized_ids = dag.get_finalized_chain()
            result[shard_id] = [dag.nodes[nid] for nid in finalized_ids if nid in dag.nodes]
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all shards"""
        stats = {
            "global_model_history_length": len(self.global_model_history),
            "shard_stats": {}
        }
        
        for shard_id, dag in self.shard_dags.items():
            stats["shard_stats"][shard_id] = dag.get_stats()
        
        return stats
