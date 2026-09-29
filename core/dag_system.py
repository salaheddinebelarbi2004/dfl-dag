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
import logging
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
    trust_score: float = 1.0
    
    def get_hash(self) -> str:
        """Get cryptographic hash of this node"""
        content = f"{self.client_id}:{self.round_num}:{self.model_hash}:{self.gradient_hash}:{self.parent_id}"
        return hashlib.sha256(content.encode()).hexdigest()


# Helper utilities
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def extract_vector(model_hash: str, dim: int = 32) -> List[float]:
    """Deterministically extract a numeric vector from a model_hash string.

    If the string already encodes a vector (comma-separated floats), parse it.
    Otherwise convert the hash bytes into a pseudo-random but deterministic vector.
    """
    if model_hash is None:
        return [0.0] * dim

    # try parse comma-separated floats
    try:
        parts = [float(x) for x in model_hash.split(",")]
        if len(parts) >= dim:
            return parts[:dim]
        # pad if short
        parts = parts + [0.0] * (dim - len(parts))
        return parts
    except Exception:
        pass

    # fallback: deterministic vector from sha256 bytes
    h = hashlib.sha256(model_hash.encode()).digest()
    # repeat bytes to reach dim
    bytes_needed = dim
    rep = (h * ((bytes_needed // len(h)) + 1))[:bytes_needed]
    vec = np.frombuffer(rep, dtype=np.uint8).astype(np.float32)
    # normalize to [-1,1]
    vec = (vec / 255.0) * 2.0 - 1.0
    # if dim differs, resize
    if len(vec) != dim:
        vec = np.resize(vec, dim)
    return vec.tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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

        # Compute trust score and decide
        try:
            trust = self.compute_trust_score(node_id)
        except Exception as e:
            logger.warning(f"Failed to compute trust for {node_id}: {e}")
            trust = 0.0

        # Store trust score
        node.trust_score = float(trust)
        logger.info(f"Node {node_id} trust_score={node.trust_score:.4f}")

        if node.trust_score < 0.3:
            node.status = UpdateStatus.REJECTED
            node.metadata["rejection_reason"] = "low trust score"
            logger.warning(f"Node {node_id} rejected: low trust score {node.trust_score:.4f}")
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

    def cross_sibling_agreement(self, parent_id: str) -> dict:
        """Compute cross-sibling agreement for children of a parent.

        Returns a dict with parent_id, sibling_scores (deviation), outliers list, mean_similarity
        """
        siblings = self.get_siblings(parent_id)
        result = {
            "parent_id": parent_id,
            "sibling_scores": {},
            "outliers": [],
            "mean_similarity": 0.0,
        }

        if not siblings:
            return result

        vectors = {}
        for sid in siblings:
            node = self.nodes.get(sid)
            if not node:
                continue
            source = node.model_hash or node.gradient_hash
            vec = extract_vector(str(source))
            vectors[sid] = np.array(vec, dtype=np.float32)

        ids = list(vectors.keys())
        if len(ids) == 0:
            return result

        # pairwise similarities
        sims = []
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a = vectors[ids[i]]
                b = vectors[ids[j]]
                s = cosine_similarity(a, b)
                sims.append(s)

        mean_sim = float(np.mean(sims)) if sims else 0.0
        result["mean_similarity"] = mean_sim

        # centroid
        centroid = np.mean(np.stack([vectors[i] for i in ids]), axis=0)
        # compute deviations: 1 - cosine_similarity(vec, centroid)
        deviations = {}
        for nid in ids:
            dev = 1.0 - cosine_similarity(vectors[nid], centroid)
            deviations[nid] = float(dev)

        dev_vals = np.array(list(deviations.values()))
        dev_mean = float(np.mean(dev_vals))
        dev_std = float(np.std(dev_vals))

        # outliers: deviation > mean + 2*std (statistical rule)
        outliers = [nid for nid, d in deviations.items() if d > (dev_mean + 2 * dev_std)]

        result["sibling_scores"] = deviations
        result["outliers"] = outliers

        if outliers:
            logger.warning(f"Outliers detected under parent {parent_id}: {outliers}")

        return result

    def compute_trust_score(self, node_id: str) -> float:
        """Compute a trust score in [0,1] for a node combining verification, structure, and agreement."""
        if node_id not in self.nodes:
            return 0.0

        node = self.nodes[node_id]

        # Verification score (0..1)
        v_flags = 0
        if node.zkp_verified:
            v_flags += 1
        if node.commitment_verified:
            v_flags += 1
        verification_score = v_flags / 2.0

        # DAG structure consistency: combine parent trust and sibling count
        parent_strength = 1.0
        if node.parent_id and node.parent_id in self.nodes:
            parent_strength = float(self.nodes[node.parent_id].trust_score)

        siblings = self.get_siblings(node.parent_id) if node.parent_id else []
        num_siblings = len(siblings)
        # sibling_factor: penalize extremely large sibling sets
        if num_siblings <= 1:
            sibling_factor = 1.0
        else:
            sibling_factor = max(0.0, 1.0 - (num_siblings - 1) / max(self.max_concurrent_siblings, num_siblings))

        structure_score = 0.5 * parent_strength + 0.5 * sibling_factor

        # Cross-sibling agreement
        agreement_score = 1.0
        try:
            if node.parent_id:
                agree = self.cross_sibling_agreement(node.parent_id)
                dev = agree.get("sibling_scores", {}).get(node_id, None)
                if dev is None:
                    agreement_score = 1.0
                else:
                    # deviation 0 -> score 1; cap between 0 and 1
                    agreement_score = max(0.0, min(1.0, 1.0 - float(dev)))
        except Exception as e:
            logger.warning(f"cross_sibling_agreement failed for {node_id}: {e}")
            agreement_score = 1.0

        # Weighted combination: 40% agreement, 30% structure, 30% verification
        total = 0.4 * agreement_score + 0.3 * structure_score + 0.3 * verification_score
        total = max(0.0, min(1.0, total))

        logger.info(f"Trust components for {node_id}: agreement={agreement_score:.3f}, structure={structure_score:.3f}, verification={verification_score:.3f} -> trust={total:.3f}")

        return float(total)
    
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
            # Filter out low-trust nodes (<0.3)
            filtered = []
            for nid in finalized_ids:
                if nid not in dag.nodes:
                    continue
                node = dag.nodes[nid]
                # root node may not have trust_score or should be included
                if nid == "root":
                    filtered.append(node)
                    continue
                if getattr(node, "trust_score", 1.0) >= 0.3:
                    filtered.append(node)
                else:
                    logger.warning(f"Excluding low-trust node from aggregation: shard={shard_id} node={nid} trust={node.trust_score}")

            result[shard_id] = filtered
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
