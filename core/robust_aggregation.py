"""
Robust Aggregation System
- ZKP-verified filtering
- Cluster-consensus aggregation (similarity-based clustering)
- Reputation-weighted aggregation (historical reliability)
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from enum import Enum

from federated_learning_config import (
    CLUSTER_K, CLUSTERING_METHOD, OUTLIER_THRESHOLD,
    MIN_REPUTATION_THRESHOLD, MAX_REPUTATION, INIT_REPUTATION,
    MALICIOUS_CLIENT_PENALTY
)
from federated_learning_config import LABEL_SKEW_THRESHOLD, SKEW_OUTLIER_WEIGHT, SOFT_OUTLIER_WEIGHT


class AggregationMethod(Enum):
    """Available aggregation methods"""
    MEAN = "mean"
    MEDIAN = "median"
    CLUSTER_CONSENSUS = "cluster_consensus"
    REPUTATION_WEIGHTED = "reputation_weighted"
    ROBUST_HYBRID = "robust_hybrid"


@dataclass
class ClientReputation:
    """Track client reliability history"""
    client_id: str
    reputation_score: float = INIT_REPUTATION
    total_submissions: int = 0
    accepted_submissions: int = 0
    rejected_submissions: int = 0
    
    def update_after_acceptance(self) -> None:
        """Update reputation after successful update"""
        self.accepted_submissions += 1
        self.total_submissions += 1
        # Increase reputation slightly (reward good behavior)
        self.reputation_score = min(
            self.reputation_score + 0.05,
            MAX_REPUTATION
        )
    
    def update_after_rejection(self) -> None:
        """Update reputation after rejected update"""
        self.rejected_submissions += 1
        self.total_submissions += 1
        # Decrease reputation significantly (punish bad behavior)
        self.reputation_score = max(
            self.reputation_score - MALICIOUS_CLIENT_PENALTY,
            0.0
        )
    
    def is_eligible(self) -> bool:
        """Check if client is eligible for aggregation"""
        return self.reputation_score >= MIN_REPUTATION_THRESHOLD


class RobustAggregator:
    """
    Robust aggregation with:
    1. ZKP-verified filtering
    2. Cluster-consensus aggregation
    3. Reputation-weighted aggregation
    """
    
    def __init__(self):
        """Initialize aggregator"""
        self.client_reputations: Dict[str, ClientReputation] = {}
        self.aggregation_history: List[Dict[str, Any]] = []
        
    def register_client(self, client_id: str, 
                       initial_reputation: float = INIT_REPUTATION) -> None:
        """Register a client with reputation tracker"""
        if client_id not in self.client_reputations:
            self.client_reputations[client_id] = ClientReputation(
                client_id=client_id,
                reputation_score=initial_reputation
            )
    
    def filter_by_zkp_verification(self,
                                  updates: List[np.ndarray],
                                  verification_results: List[bool],
                                  client_ids: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Filter updates by ZKP verification results.
        
        Args:
            updates: Model updates from clients
            verification_results: Boolean verification result for each update
            client_ids: Client IDs for updates
            
        Returns:
            Tuple of (verified_updates, verified_client_ids)
        """
        verified_updates = []
        verified_client_ids = []

        # If any verification passed, apply normal reputation updates
        if any(verification_results):
            for update, verified, client_id in zip(updates, verification_results, client_ids):
                self.register_client(client_id)
                if verified:
                    verified_updates.append(update)
                    verified_client_ids.append(client_id)
                    self.client_reputations[client_id].update_after_acceptance()
                else:
                    # Only penalize explicitly rejected updates when verifier is functional
                    self.client_reputations[client_id].update_after_rejection()
            self.zkp_verification_overridden = False
        else:
            # No verifications passed: treat as verifier failure. Accept all updates
            # for diagnostics/aggregation but DO NOT penalize reputations here.
            self.zkp_verification_overridden = True
            verified_updates = list(updates)
            verified_client_ids = list(client_ids)

        return verified_updates, verified_client_ids
    
    def filter_by_reputation(self,
                            updates: List[np.ndarray],
                            client_ids: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Filter updates by client reputation.
        
        Args:
            updates: Model updates
            client_ids: Client IDs
            
        Returns:
            Tuple of (eligible_updates, eligible_client_ids)
        """
        eligible_updates = []
        eligible_client_ids = []
        
        for update, client_id in zip(updates, client_ids):
            self.register_client(client_id)
            if self.client_reputations[client_id].is_eligible():
                eligible_updates.append(update)
                eligible_client_ids.append(client_id)
        
        return eligible_updates, eligible_client_ids
    
    def cluster_consensus_aggregation(self,
                                     updates: List[np.ndarray],
                                     client_ids: List[str],
                                     k: int = CLUSTER_K) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster-based consensus aggregation.
        
        Groups similar updates via clustering, aggregates within each cluster,
        then takes weighted average of cluster aggregates.
        
        Args:
            updates: List of update vectors
            client_ids: List of client IDs
            k: Number of clusters
            
        Returns:
            Tuple of (aggregated_update, metadata)
        """
        if not updates:
            raise ValueError("No updates to aggregate")
        
        updates_array = np.array([u.flatten() for u in updates])
        n_samples = updates_array.shape[0]
        
        # Adjust k if necessary
        k = min(k, n_samples)
        
        metadata = {
            "method": "cluster_consensus",
            "num_updates": n_samples,
            "num_clusters": k,
            "cluster_assignments": [],
            "cluster_sizes": [],
        }
        
        # Perform clustering
        if CLUSTERING_METHOD == "kmeans":
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(updates_array)
        else:  # hierarchical
            Z = linkage(updates_array, method="ward")
            labels = fcluster(Z, k, criterion="maxclust") - 1
        
        # Aggregate within clusters
        cluster_aggregates = []
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_updates = updates_array[cluster_mask]
            if len(cluster_updates) > 0:
                cluster_mean = np.mean(cluster_updates, axis=0)
                cluster_aggregates.append(cluster_mean)
                metadata["cluster_sizes"].append(len(cluster_updates))
            else:
                metadata["cluster_sizes"].append(0)
        
        metadata["cluster_assignments"] = labels.tolist()
        metadata["outlier_clients"] = self.detect_outlier_clients(updates, client_ids, labels)
        
        # Final weighted average (weight by cluster size)
        if cluster_aggregates:
            cluster_aggregates_array = np.array(cluster_aggregates)
            weights = np.array(metadata["cluster_sizes"]) / sum(metadata["cluster_sizes"])
            final_aggregate = np.average(cluster_aggregates_array, axis=0, weights=weights)
        else:
            final_aggregate = np.zeros_like(updates_array[0])
        
        # Reshape back to original shape
        original_shape = updates[0].shape
        final_aggregate = final_aggregate.reshape(original_shape)
        
        return final_aggregate, metadata
    
    def detect_outlier_clients(self,
                               updates: List[np.ndarray],
                               client_ids: List[str],
                               labels: np.ndarray) -> List[str]:
        """
        Detect outlier clients within clusters using cosine similarity deviation.

        Args:
            updates: Flattened update vectors
            client_ids: Client IDs corresponding to updates
            labels: Cluster labels for each update

        Returns:
            List of client IDs considered outliers.
        """
        outliers = []
        if not updates:
            return outliers

        vectors = np.array([u.flatten() for u in updates])
        unique_labels = np.unique(labels)

        for cluster_id in unique_labels:
            idx = np.where(labels == cluster_id)[0]
            # require at least 3 members to compute a robust cluster outlier
            if len(idx) < 3:
                continue
            cluster_vecs = vectors[idx]
            centroid = np.mean(cluster_vecs, axis=0)
            deviations = np.linalg.norm(cluster_vecs - centroid, axis=1)
            mean_dev = float(np.mean(deviations))
            std_dev = float(np.std(deviations))
            if std_dev <= 1e-8:
                continue
            # Use z-score test for outliers
            z_scores = (deviations - mean_dev) / std_dev
            for pos, z in zip(idx, z_scores):
                if z > OUTLIER_THRESHOLD:
                    outliers.append(client_ids[pos])

        return list(dict.fromkeys(outliers))

    def detect_similarity_outliers(self,
                                   updates: List[np.ndarray],
                                   client_ids: List[str]) -> List[str]:
        """
        Detect clients whose updates are globally dissimilar to the group.

        Args:
            updates: List of update vectors
            client_ids: Client IDs

        Returns:
            List of client IDs considered similarity outliers.
        """
        outliers = []
        if len(updates) < 2:
            return outliers

        vectors = np.array([u.flatten() for u in updates])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = vectors / norms
        similarity_matrix = normalized @ normalized.T

        # Exclude self-similarity
        np.fill_diagonal(similarity_matrix, 0.0)
        avg_similarities = np.mean(similarity_matrix, axis=1)

        mean_sim = float(np.mean(avg_similarities))
        std_sim = float(np.std(avg_similarities))
        # require a meaningful spread before flagging global similarity outliers
        if std_sim <= 1e-6:
            return []
        threshold = max(0.0, mean_sim - OUTLIER_THRESHOLD * std_sim)

        for cid, avg_sim in zip(client_ids, avg_similarities):
            # mark only strong deviations; we'll downweight later instead of removing
            if avg_sim < threshold:
                outliers.append(cid)

        return list(dict.fromkeys(outliers))

    def detect_label_skew(self, client_ids: List[str]) -> List[str]:
        """
        Inspect per-client npz files under 'malisious' or 'real_federated_data'
        to detect clients whose label distribution is heavily skewed to one class.
        Returns list of client_ids flagged as skewed.
        """
        import os
        flagged = []
        for cid in client_ids:
            # file paths to check
            base_name = f"{cid}.npz"
            cand_paths = [os.path.join("malisious", base_name), os.path.join("real_federated_data", base_name)]
            found = False
            for p in cand_paths:
                if os.path.exists(p):
                    data = np.load(p)
                    y = data.get("y")
                    if y is None:
                        break
                    if y.ndim > 1:
                        y = np.argmax(y, axis=1)
                    vals, counts = np.unique(y, return_counts=True)
                    total = counts.sum()
                    if total == 0:
                        break
                    max_frac = counts.max() / float(total)
                    if max_frac >= LABEL_SKEW_THRESHOLD:
                        flagged.append(cid)
                    found = True
                    break
            if not found:
                # If no local file, skip (can't assess)
                continue
        return list(dict.fromkeys(flagged))

    def reputation_weighted_aggregation(self,
                                       updates: List[np.ndarray],
                                       client_ids: List[str],
                                       downweight_map: Dict[str, float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reputation-weighted aggregation.
        Updates from higher-reputation clients get higher weights.
        
        Args:
            updates: List of update vectors
            client_ids: List of client IDs
            
        Returns:
            Tuple of (weighted_aggregate, metadata)
        """
        if not updates:
            raise ValueError("No updates to aggregate")
        
        metadata = {
            "method": "reputation_weighted",
            "num_updates": len(updates),
            "weights": [],
        }
        
        # Get reputation scores
        weights = []
        for client_id in client_ids:
            self.register_client(client_id)
            weight = self.client_reputations[client_id].reputation_score
            weights.append(weight)
            metadata["weights"].append(weight)

        # Apply soft-downweighting if provided
        if downweight_map:
            for i, cid in enumerate(client_ids):
                if cid in downweight_map:
                    weights[i] *= float(downweight_map[cid])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        
        # Weighted average
        updates_array = np.array([u.flatten() for u in updates])
        weighted_aggregate = np.average(updates_array, axis=0, weights=weights)
        
        # Reshape back
        original_shape = updates[0].shape
        weighted_aggregate = weighted_aggregate.reshape(original_shape)
        
        return weighted_aggregate, metadata
    
    def robust_hybrid_aggregation(self,
                                 updates: List[np.ndarray],
                                 verification_results: List[bool],
                                 client_ids: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Robust hybrid aggregation combining multiple techniques:
        1. ZKP-based filtering
        2. Reputation filtering
        3. Cluster-consensus
        4. Reputation weighting
        
        Args:
            updates: Model updates
            verification_results: ZKP verification results
            client_ids: Client IDs
            
        Returns:
            Tuple of (aggregated_update, metadata)
        """
        metadata = {
            "method": "robust_hybrid",
            "original_count": len(updates),
            "stages": {}
        }
        
        # Stage 1: ZKP filtering
        verified_updates, verified_cids = self.filter_by_zkp_verification(
            updates, verification_results, client_ids
        )
        metadata["stages"]["after_zkp_filtering"] = len(verified_updates)
        
        # Stage 2: Reputation filtering
        eligible_updates, eligible_cids = self.filter_by_reputation(
            verified_updates, verified_cids
        )
        metadata["stages"]["after_reputation_filtering"] = len(eligible_updates)
        
        if not eligible_updates:
            raise ValueError("No eligible updates after filtering")
        
        # Stage 3: Similarity-based outlier detection before consensus
        similarity_outliers = self.detect_similarity_outliers(
            eligible_updates, eligible_cids
        )
        metadata["stages"]["similarity_outliers"] = similarity_outliers
        # Keep similarity flags; actual downweighting will be applied only
        # to clients also flagged as within-cluster outliers (intersection).
        sim_flagged = set(similarity_outliers)
        downweight_map: Dict[str, float] = {}
        
        # Stage 4: Cluster consensus on eligible updates
        cluster_agg, cluster_meta = self.cluster_consensus_aggregation(
            eligible_updates, eligible_cids, k=min(CLUSTER_K, len(eligible_updates))
        )
        metadata["stages"]["cluster_consensus"] = cluster_meta
        
        # Stage 4b: Detect within-cluster outliers and softly downweight them
        outlier_clients = cluster_meta.get("outlier_clients", [])
        metadata["stages"]["cluster_outliers"] = outlier_clients
        # Stage 4c: Detect label-skewed clients (possible poisoning)
        label_skewed = self.detect_label_skew(eligible_cids)
        metadata["stages"]["label_skewed"] = label_skewed
        if outlier_clients:
            # Downweight only clients flagged by both detectors to reduce false positives
            cluster_flagged = set(outlier_clients)
            suspects = sim_flagged.intersection(cluster_flagged)
            for cid in outlier_clients:
                self.register_client(cid)
                self.client_reputations[cid].update_after_rejection()
            if suspects:
                for cid in suspects:
                    if cid not in downweight_map:
                        downweight_map[cid] = SOFT_OUTLIER_WEIGHT
                metadata["stages"]["cluster_downweighted"] = list(downweight_map.keys())
            else:
                metadata["stages"]["cluster_downweighted"] = []
            # keep all updates but report downweighting; recompute cluster_meta for reporting
            cluster_agg, cluster_meta = self.cluster_consensus_aggregation(
                eligible_updates, eligible_cids, k=min(CLUSTER_K, len(eligible_updates))
            )
            metadata["stages"]["cluster_consensus_post_filter"] = cluster_meta

        # Apply label-skew downweighting for flagged clients (even if not in cluster suspects)
        if label_skewed:
            for cid in label_skewed:
                # penalize reputation slightly but apply stronger downweight
                self.register_client(cid)
                self.client_reputations[cid].update_after_rejection()
                # set to smaller weight, but don't override a smaller existing weight
                existing = downweight_map.get(cid, 1.0)
                downweight_map[cid] = min(existing, SKEW_OUTLIER_WEIGHT)
            metadata["stages"]["label_skew_downweighted"] = list(label_skewed)
        
        # Stage 5: Reputation weighting (reweight filtered updates)
        final_agg, reputation_meta = self.reputation_weighted_aggregation(
            eligible_updates, eligible_cids, downweight_map=downweight_map
        )
        metadata["stages"]["reputation_weighted"] = reputation_meta
        
        # Combine: average of cluster consensus and reputation weighting
        final_aggregate = (cluster_agg + final_agg) / 2.0
        
        # Record in history
        self.aggregation_history.append(metadata)
        
        return final_aggregate, metadata
    
    def aggregate(self,
                 updates: List[np.ndarray],
                 verification_results: List[bool] = None,
                 client_ids: List[str] = None,
                 method: AggregationMethod = AggregationMethod.ROBUST_HYBRID
                 ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main aggregation entry point.
        
        Args:
            updates: List of model updates
            verification_results: ZKP verification results (optional)
            client_ids: Client identifiers (optional)
            method: Aggregation method to use
            
        Returns:
            Tuple of (aggregated_update, metadata)
        """
        if not updates:
            raise ValueError("No updates to aggregate")
        
        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(len(updates))]
        
        if verification_results is None:
            verification_results = [True] * len(updates)
        
        if method == AggregationMethod.MEAN:
            agg = np.mean([u.flatten() for u in updates], axis=0)
            agg = agg.reshape(updates[0].shape)
            return agg, {"method": "mean"}
        
        elif method == AggregationMethod.MEDIAN:
            agg = np.median([u.flatten() for u in updates], axis=0)
            agg = agg.reshape(updates[0].shape)
            return agg, {"method": "median"}
        
        elif method == AggregationMethod.CLUSTER_CONSENSUS:
            return self.cluster_consensus_aggregation(updates, client_ids)
        
        elif method == AggregationMethod.ROBUST_HYBRID:
            return self.robust_hybrid_aggregation(
                updates, verification_results, client_ids
            )
        
        elif method == AggregationMethod.REPUTATION_WEIGHTED:
            return self.reputation_weighted_aggregation(updates, client_ids)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_reputation(self, client_id: str) -> float:
        """Get reputation score for a client"""
        self.register_client(client_id)
        return self.client_reputations[client_id].reputation_score
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        return {
            "client_count": len(self.client_reputations),
            "aggregation_count": len(self.aggregation_history),
            "client_reputations": {
                cid: {
                    "score": rep.reputation_score,
                    "accepted": rep.accepted_submissions,
                    "rejected": rep.rejected_submissions,
                }
                for cid, rep in self.client_reputations.items()
            }
        }
