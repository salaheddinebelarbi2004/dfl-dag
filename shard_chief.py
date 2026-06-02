"""
Shard Chief System
- Score-based election: Sj = α·Uj + β·Bj + γ·Cj
- Top-K candidates → random uniform selection for rotation
- Verification and robust aggregation
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import random
import time

from federated_learning_config import (
    SHARD_CHIEF_SCORE_ALPHA, SHARD_CHIEF_SCORE_BETA,
    SHARD_CHIEF_SCORE_GAMMA, TOP_K_CANDIDATES
)
from robust_aggregation import RobustAggregator, AggregationMethod
from zkp_proofs import ZKProof, ZKPVerifier, ZKPFactory, Groth16Statement

import hashlib


@dataclass
class NodeCapabilities:
    """Node capability metrics"""
    node_id: str
    uptime: float = 0.95  # Fraction of time available
    bandwidth_mbps: float = 100.0  # Bandwidth in Mbps
    compute_capacity: float = 1.0  # Relative compute (1.0 = baseline)
    
    def get_uptime_score(self) -> float:
        """Normalize uptime to [0, 1]"""
        return min(self.uptime, 1.0)
    
    def get_bandwidth_score(self) -> float:
        """Normalize bandwidth to [0, 1]"""
        # Assume baseline of 100 Mbps
        return min(self.bandwidth_mbps / 100.0, 1.0)
    
    def get_compute_score(self) -> float:
        """Get compute capacity score"""
        return min(self.compute_capacity, 1.0)


@dataclass
class ShardChiefCandidate:
    """Candidate for Shard Chief position"""
    node_id: str
    capabilities: NodeCapabilities
    score: float = 0.0
    rank: int = -1
    
    def compute_score(self, alpha: float = SHARD_CHIEF_SCORE_ALPHA,
                     beta: float = SHARD_CHIEF_SCORE_BETA,
                     gamma: float = SHARD_CHIEF_SCORE_GAMMA) -> float:
        """
        Compute Shard Chief score.
        Sj = α·Uj + β·Bj + γ·Cj
        
        Args:
            alpha: Weight for uptime
            beta: Weight for bandwidth
            gamma: Weight for compute
            
        Returns:
            Score in [0, 1]
        """
        uptime_score = self.capabilities.get_uptime_score()
        bandwidth_score = self.capabilities.get_bandwidth_score()
        compute_score = self.capabilities.get_compute_score()
        
        self.score = (
            alpha * uptime_score +
            beta * bandwidth_score +
            gamma * compute_score
        )
        return self.score


@dataclass
class ShardChief:
    """
    Shard Chief - manages shard aggregation and verification
    """
    chief_id: str
    shard_id: str
    capabilities: NodeCapabilities
    
    # Time tracking
    elected_at: float = field(default_factory=time.time)
    term_duration: float = 3600.0  # 1 hour term
    
    # Aggregation
    aggregator: RobustAggregator = field(default_factory=RobustAggregator)
    
    # Verification
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistics
    updates_verified: int = 0
    updates_rejected: int = 0
    aggregations_performed: int = 0
    
    def is_active(self) -> bool:
        """Check if chief is still active in term"""
        elapsed = time.time() - self.elected_at
        return elapsed < self.term_duration
    
    def verify_updates(self,
                      gradients: List[np.ndarray],
                      proofs: List[ZKProof],
                      client_ids: List[str]) -> Tuple[List[bool], List[Dict]]:
        """
        Verify ZKP proofs for all client updates.
        
        Args:
            gradients: Gradient updates
            proofs: ZKP proofs
            client_ids: Client identifiers
            
        Returns:
            Tuple of (verification_results, verification_details)
        """
        verifier = ZKPFactory.get_verifier()
        results = []
        details = []
        
        for gradient, proof, client_id in zip(gradients, proofs, client_ids):
            public_statement = None
            if hasattr(proof, "metadata") and isinstance(proof.metadata, dict):
                statement_data = proof.metadata.get("public_statement")
                if isinstance(statement_data, dict) and statement_data:
                    public_statement = Groth16Statement(
                        one=statement_data.get("one", 1),
                        h_w_old=statement_data.get("h_w_old", ""),
                        h_w_new=statement_data.get("h_w_new", ""),
                        model_hash=statement_data.get("model_hash", ""),
                        round_num=statement_data.get("round_num", 0),
                        shard_id=statement_data.get("shard_id", self.shard_id),
                        public_inputs=statement_data.get("public_inputs", []),
                    )

            is_valid, detail = verifier.verify_proof(
                proof, gradient,
                verifier_id=self.chief_id,
                statement=public_statement,
            )
            results.append(is_valid)
            detail["client_id"] = client_id
            detail["protocol"] = getattr(proof, "metadata", {}).get("protocol", "Groth16") if hasattr(proof, "metadata") else "Groth16"
            details.append(detail)
            
            if is_valid:
                self.updates_verified += 1
            else:
                self.updates_rejected += 1
        
        self.verification_results.extend(details)
        return results, details
    
    def aggregate_updates(self,
                         gradients: List[np.ndarray],
                         verification_results: List[bool],
                         client_ids: List[str],
                         model_params: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        Perform robust aggregation of verified updates.
        
        Uses hybrid robust aggregation:
        1. ZKP filtering
        2. Reputation filtering
        3. Cluster consensus
        4. Reputation weighting
        
        Args:
            gradients: Client gradient updates
            verification_results: ZKP verification results
            client_ids: Client identifiers
            model_params: Current model parameters (optional)
            
        Returns:
            Tuple of (aggregated_update, metadata)
        """
        # Use robust hybrid aggregation
        aggregated, metadata = self.aggregator.aggregate(
            gradients,
            verification_results=verification_results,
            client_ids=client_ids,
            method=AggregationMethod.ROBUST_HYBRID
        )
        
        self.aggregations_performed += 1
        metadata["chief_id"] = self.chief_id
        metadata["shard_id"] = self.shard_id
        
        return aggregated, metadata
    
    def process_round(self,
                     gradients: List[np.ndarray],
                     proofs: List[ZKProof],
                     client_ids: List[str],
                     model_params: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        Full round processing: verify → aggregate.
        
        Implements Step 7-9 of training workflow.
        
        Args:
            gradients: Client updates
            proofs: ZKP proofs
            client_ids: Client IDs
            model_params: Current model
            
        Returns:
            Tuple of (shard_aggregate, process_metadata)
        """
        metadata = {
            "chief_id": self.chief_id,
            "shard_id": self.shard_id,
            "round_timestamp": time.time(),
        }
        
        # Step 7: Verify ZKP proofs
        verification_results, verification_details = self.verify_updates(
            gradients, proofs, client_ids
        )
        metadata["verification"] = {
            "results": verification_results,
            "summary": f"{sum(verification_results)}/{len(verification_results)} verified",
        }
        
        # Step 8: Robust aggregation with filtering
        try:
            aggregated, agg_metadata = self.aggregate_updates(
                gradients, verification_results, client_ids, model_params
            )
            metadata["aggregation"] = agg_metadata
            
            # Step 9: Shard model update
            metadata["shard_model"] = {
                "aggregation_successful": True,
                "updates_used": agg_metadata["stages"]["after_reputation_filtering"],
                "total_received": len(gradients),
            }
            
            return aggregated, metadata
        
        except Exception as e:
            metadata["aggregation_error"] = str(e)
            metadata["shard_model"] = {"aggregation_successful": False, "error": str(e)}
            return np.zeros_like(gradients[0]), metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Shard Chief statistics"""
        return {
            "chief_id": self.chief_id,
            "shard_id": self.shard_id,
            "is_active": self.is_active(),
            "updates_verified": self.updates_verified,
            "updates_rejected": self.updates_rejected,
            "aggregations_performed": self.aggregations_performed,
            "verification_success_rate": (
                self.updates_verified / (self.updates_verified + self.updates_rejected)
                if (self.updates_verified + self.updates_rejected) > 0
                else 0
            ),
            "capabilities": {
                "uptime": self.capabilities.uptime,
                "bandwidth_mbps": self.capabilities.bandwidth_mbps,
                "compute_capacity": self.capabilities.compute_capacity,
            }
        }


class ShardChiefElector:
    """
    Elects Shard Chiefs from node pool using score-based ranking
    """
    
    @staticmethod
    def elect_chief(candidates: List[ShardChiefCandidate],
                   shard_id: str,
                   top_k: int = TOP_K_CANDIDATES) -> ShardChief:
        """
        Elect a Shard Chief from candidates.
        
        Process:
        1. Score all candidates: Sj = α·Uj + β·Bj + γ·Cj
        2. Select top K
        3. Randomly choose one from top K (for rotation)
        
        Args:
            candidates: List of candidate nodes
            shard_id: Shard ID
            top_k: Number of top candidates to consider
            
        Returns:
            Elected ShardChief
        """
        # Step 1: Compute scores
        for candidate in candidates:
            candidate.compute_score()
        
        # Step 2: Sort by score
        candidates_sorted = sorted(candidates, key=lambda c: c.score, reverse=True)
        
        # Step 3: Select top K
        top_k = min(top_k, len(candidates_sorted))
        top_k_candidates = candidates_sorted[:top_k]
        
        # Step 4: Assign ranks
        for i, candidate in enumerate(top_k_candidates):
            candidate.rank = i + 1
        
        # Step 5: Random uniform selection from top K (for rotation)
        elected = random.choice(top_k_candidates)
        
        # Create ShardChief instance
        chief = ShardChief(
            chief_id=elected.node_id,
            shard_id=shard_id,
            capabilities=elected.capabilities,
        )
        
        return chief, {
            "elected_node": elected.node_id,
            "score": elected.score,
            "rank": elected.rank,
            "top_k": top_k,
            "total_candidates": len(candidates),
        }
    
    @staticmethod
    def elect_for_shard(all_candidates: List[ShardChiefCandidate],
                       shard_id: str,
                       shard_local_nodes: List[str] = None,
                       top_k: int = TOP_K_CANDIDATES) -> Tuple[ShardChief, Dict]:
        """
        Elect chief for a specific shard from local or global candidates.
        
        Args:
            all_candidates: All available candidate nodes
            shard_id: Shard ID
            shard_local_nodes: Nodes specific to this shard (optional)
            top_k: Top K to consider
            
        Returns:
            Tuple of (elected_chief, election_metadata)
        """
        # Filter to shard-local candidates if specified
        if shard_local_nodes:
            candidates = [
                c for c in all_candidates
                if c.node_id in shard_local_nodes
            ]
        else:
            candidates = all_candidates
        
        if not candidates:
            raise ValueError(f"No candidates for shard {shard_id}")
        
        return ShardChiefElector.elect_chief(candidates, shard_id, top_k)
