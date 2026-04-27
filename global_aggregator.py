"""
Global Aggregator
Combines shard models to produce global model update.
Implements Step 10 of training workflow.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import time
import hashlib


@dataclass
class ShardModelUpdate:
    """Model update from a shard"""
    shard_id: str
    shard_model: np.ndarray
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    num_updates_aggregated: int = 0
    verification_success_rate: float = 0.0


class GlobalAggregator:
    """
    Global layer aggregator.
    Combines shard models to produce global update.
    Implements Step 10: Wt+1 = Aggregate(Ws1, ..., Wsk)
    """

    def __init__(self, num_shards: int):
        """
        Initialize global aggregator.
        
        Args:
            num_shards: Number of shards in system
        """
        self.num_shards = num_shards
        self.global_model: Optional[np.ndarray] = None
        self.aggregation_history: List[Dict[str, Any]] = []
        self.round_num = 0

    def aggregate_shard_models(self,
                              shard_updates: List[ShardModelUpdate],
                              aggregation_method: str = "weighted_mean") -> Tuple[np.ndarray, Dict]:
        """
        Aggregate shard models to compute global model update.

        Methods:
        - "mean": Simple average of shard models
        - "weighted_mean": Weighted by number of aggregated updates per shard
        - "median": Median across shards (robust)

        Args:
            shard_updates: List of shard model updates
            aggregation_method: Method to use

        Returns:
            Tuple of (global_model_update, metadata)
        """
        if not shard_updates:
            raise ValueError("No shard updates to aggregate")

        metadata = {
            "round_num": self.round_num,
            "num_shards": len(shard_updates),
            "method": aggregation_method,
            "shard_details": [],
            "timestamp": time.time(),
        }

        models = []
        weights = []

        for shard_update in shard_updates:
            models.append(shard_update.shard_model)
            weights.append(shard_update.num_updates_aggregated)

            metadata["shard_details"].append({
                "shard_id": shard_update.shard_id,
                "updates_aggregated": shard_update.num_updates_aggregated,
                "verification_rate": shard_update.verification_success_rate,
            })

        models_array = np.array([m.flatten() for m in models])

        # Aggregate
        if aggregation_method == "mean":
            global_update = np.mean(models_array, axis=0)

        elif aggregation_method == "weighted_mean":
            weights_norm = np.array(weights, dtype=float)
            weights_norm = weights_norm / np.sum(weights_norm)
            global_update = np.average(models_array, axis=0, weights=weights_norm)

        elif aggregation_method == "median":
            global_update = np.median(models_array, axis=0)

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Reshape to original shape
        original_shape = shard_updates[0].shard_model.shape
        global_update = global_update.reshape(original_shape)

        metadata["aggregation_successful"] = True
        self.aggregation_history.append(metadata)

        return global_update, metadata

    def update_global_model(self,
                           current_global_model: np.ndarray,
                           global_update: np.ndarray,
                           learning_rate: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Apply global update to current model.
        Wt+1 = Wt + α * ∆W_global

        Args:
            current_global_model: Current global model Wt
            global_update: Aggregated update ∆W_global
            learning_rate: Global learning rate

        Returns:
            Tuple of (new_global_model, update_metadata)
        """
        new_global_model = current_global_model + learning_rate * global_update

        metadata = {
            "learning_rate": learning_rate,
            "update_norm": float(np.linalg.norm(global_update.flatten())),
            "model_change_norm": float(
                np.linalg.norm((new_global_model - current_global_model).flatten())
            ),
        }

        return new_global_model, metadata

    def aggregate_round(self,
                       current_global_model: np.ndarray,
                       shard_updates: List[ShardModelUpdate],
                       global_learning_rate: float = 1.0,
                       aggregation_method: str = "weighted_mean") -> Tuple[np.ndarray, Dict]:
        """
        Complete global aggregation for one round.

        Implements Step 10: Wt+1 = Aggregate(Ws)

        Args:
            current_global_model: Current global model Wt
            shard_updates: Model updates from all shards
            global_learning_rate: Global learning rate
            aggregation_method: Aggregation method

        Returns:
            Tuple of (new_global_model, round_metadata)
        """
        self.round_num += 1

        round_metadata = {
            "round_num": self.round_num,
            "current_model_hash": self._hash_model(current_global_model),
            "timestamp": time.time(),
        }

        try:
            # Step 10a: Aggregate shard models
            global_update, agg_meta = self.aggregate_shard_models(
                shard_updates, aggregation_method=aggregation_method
            )
            round_metadata["aggregation"] = agg_meta

            # Step 10b: Update global model
            new_global_model, update_meta = self.update_global_model(
                current_global_model, global_update, learning_rate=global_learning_rate
            )
            round_metadata["model_update"] = update_meta

            # Verify update
            round_metadata["new_model_hash"] = self._hash_model(new_global_model)
            round_metadata["success"] = True

            self.global_model = new_global_model

            return new_global_model, round_metadata

        except Exception as e:
            round_metadata["success"] = False
            round_metadata["error"] = str(e)
            return current_global_model, round_metadata

    def _hash_model(self, model: np.ndarray) -> str:
        """Compute model hash"""
        model_bytes = model.astype(np.float32).tobytes()
        return hashlib.sha256(model_bytes).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        successful_rounds = sum(1 for h in self.aggregation_history if h.get("success", False))

        return {
            "num_shards": self.num_shards,
            "round_num": self.round_num,
            "total_aggregations": len(self.aggregation_history),
            "successful_aggregations": successful_rounds,
            "aggregation_success_rate": (
                successful_rounds / len(self.aggregation_history)
                if self.aggregation_history else 0
            ),
        }


class GlobalCoordinator:
    """
    Coordinates the entire system:
    - Broadcasts global model to shards
    - Collects shard updates
    - Performs global aggregation
    - Manages training rounds
    """

    def __init__(self, num_shards: int, initial_model: np.ndarray):
        """
        Initialize coordinator.
        
        Args:
            num_shards: Number of shards
            initial_model: Initial global model
        """
        self.num_shards = num_shards
        self.global_model = initial_model.copy()
        self.global_aggregator = GlobalAggregator(num_shards)
        self.round_num = 0

        # Communication tracking
        self.broadcast_history: List[Dict] = []
        self.collection_history: List[Dict] = []

    def broadcast_global_model(self) -> Dict[str, Any]:
        """
        Broadcast current global model to all shards.
        Implements Step 1 of training workflow.

        Returns:
            Broadcast metadata
        """
        broadcast_meta = {
            "round_num": self.round_num,
            "timestamp": time.time(),
            "model_hash": self._hash_model(self.global_model),
            "model_shape": self.global_model.shape,
            "num_shards_receiving": self.num_shards,
        }

        self.broadcast_history.append(broadcast_meta)
        return broadcast_meta

    def collect_shard_updates(self,
                             shard_updates: List[ShardModelUpdate]) -> Dict[str, Any]:
        """
        Collect shard model updates.
        Called after shards complete their aggregation.

        Args:
            shard_updates: Updates from all shards

        Returns:
            Collection metadata
        """
        collection_meta = {
            "round_num": self.round_num,
            "timestamp": time.time(),
            "num_updates_received": len(shard_updates),
            "expected_updates": self.num_shards,
            "complete": len(shard_updates) == self.num_shards,
        }

        self.collection_history.append(collection_meta)
        return collection_meta

    def run_training_round(self,
                          shard_updates: List[ShardModelUpdate],
                          global_learning_rate: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Run one complete training round.

        Implements full workflow Steps 1-10:
        1. Broadcast global model (by caller before this)
        2-6. Clients: local training + update generation
        7-9. Shards: verification + aggregation
        10. Global: aggregate shard models

        Args:
            shard_updates: Updates collected from shards
            global_learning_rate: Global learning rate

        Returns:
            Tuple of (new_global_model, round_metadata)
        """
        self.round_num += 1

        round_meta = {
            "round_num": self.round_num,
            "timestamp": time.time(),
        }

        # Broadcast (Step 1) - done by caller
        broadcast_meta = self.broadcast_global_model()
        round_meta["broadcast"] = broadcast_meta

        # Collect shard updates (after Steps 2-9)
        collection_meta = self.collect_shard_updates(shard_updates)
        round_meta["collection"] = collection_meta

        # Step 10: Global aggregation
        new_global_model, agg_meta = self.global_aggregator.aggregate_round(
            self.global_model,
            shard_updates,
            global_learning_rate=global_learning_rate
        )
        round_meta["aggregation"] = agg_meta

        # Update internal model
        if agg_meta.get("success", False):
            self.global_model = new_global_model

        return self.global_model, round_meta

    def _hash_model(self, model: np.ndarray) -> str:
        """Compute model hash"""
        model_bytes = model.astype(np.float32).tobytes()
        return hashlib.sha256(model_bytes).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            "round_num": self.round_num,
            "global_model_hash": self._hash_model(self.global_model),
            "broadcasts": len(self.broadcast_history),
            "collections": len(self.collection_history),
            "aggregator_stats": self.global_aggregator.get_stats(),
        }
