import os
import sys
import time
import json
import base64
import pickle
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "12"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "3"))
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "5"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2"))
P2P_PORT = int(os.environ.get("P2P_PORT", "5555"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
DATA_DIR = os.environ.get("DATA_DIR", "/data/real_federated_data")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")
RESULTS_SUBDIR = os.environ.get("RESULTS_SUBDIR", "")
RUNNER_MODE = os.environ.get("RUNNER_MODE", "legacy").strip().lower()
RUN_LABEL = os.environ.get("RUN_LABEL", "").strip()

CLIENT_NAME = f"client_{CLIENT_ID}"

clients_per_shard = NUM_CLIENTS // NUM_SHARDS
SHARD_ID = CLIENT_ID // clients_per_shard
if SHARD_ID >= NUM_SHARDS:
    SHARD_ID = NUM_SHARDS - 1

shard_start = SHARD_ID * clients_per_shard
shard_end = (
    shard_start + clients_per_shard if SHARD_ID < NUM_SHARDS - 1 else NUM_CLIENTS
)
SHARD_MEMBERS = list(range(shard_start, shard_end))
SHARD_CHIEF_ID = min(SHARD_MEMBERS)
IS_CHIEF = CLIENT_ID == SHARD_CHIEF_ID
IS_COORDINATOR = CLIENT_ID == 0

print(f"""
{"=" * 60}
  Docker Federated Learning Client
{"=" * 60}
  Client ID:     {CLIENT_NAME}
  Shard ID:      shard_{SHARD_ID}
  Shard Members: {SHARD_MEMBERS}
  Shard Chief:   client_{SHARD_CHIEF_ID} (me: {IS_CHIEF})
  Coordinator:   client_0 (me: {IS_COORDINATOR})
  Config:        {NUM_CLIENTS} clients, {NUM_SHARDS} shards, {NUM_ROUNDS} rounds
{"=" * 60}
""")


class DockerFLClient:
    def __init__(self):
        self.client_id = CLIENT_ID
        self.client_name = CLIENT_NAME
        self.shard_id = SHARD_ID
        self.is_chief = IS_CHIEF
        self.is_coordinator = IS_COORDINATOR
        self.shard_members = SHARD_MEMBERS
        self.shard_chief_id = SHARD_CHIEF_ID

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.model = None
        self.node: Optional[Any] = None
        self.num_train_samples = 0
        self.num_test_samples = 0

        self.round_num = 0
        self.received_updates: Dict[str, Dict] = {}
        self.shard_models: Dict[int, Dict] = {}
        self.global_weights: Optional[Dict] = None
        self.round_metrics: List[Dict] = []

        self._lock = threading.Lock()
        self._global_received = threading.Event()
        self._round_started = threading.Event()

    # ---------- DATA LOADING ----------

    def load_data(self) -> bool:
        data_path = Path(DATA_DIR) / f"client_{self.client_id}.npz"
        if not data_path.exists():
            alt_path = (
                Path(DATA_DIR.replace("real_federated_data", ""))
                / "real_federated_data"
                / f"client_{self.client_id}.npz"
            )
            data_path = alt_path
        if not data_path.exists():
            print(f"  DATA ERROR: {data_path} not found")
            return False

        data = np.load(data_path)
        X = data["X"] if "X" in data.files else data["embeddings"]
        y = data["y"] if "y" in data.files else data["labels"]

        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        else:
            y = y.flatten()

        rng = np.random.default_rng(42 + self.client_id)
        indices = rng.permutation(len(X))
        train_size = max(1, int(len(X) * 0.8))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        self.X_train = X[train_idx].astype(np.float32)
        self.y_train = y[train_idx].astype(int)
        self.X_test = X[test_idx].astype(np.float32)
        self.y_test = y[test_idx].astype(int)
        self.num_train_samples = len(self.X_train)
        self.num_test_samples = len(self.X_test)

        print(
            f"  Loaded {len(X)} samples: train={len(self.X_train)}, test={len(self.X_test)}"
        )
        return True

    def init_model(self) -> bool:
        try:
            from model_registry import registry
            from run_with_shards import RealModelWrapper

            pt_path = Path("/app/model.pt")
            safe_path = Path("/app/model.safetensors")
            if safe_path.exists():
                model_path = str(safe_path)
                loader = "safetensors"
            elif pt_path.exists():
                model_path = str(pt_path)
                loader = "pytorch"
            else:
                model_path = "model.pt"
                loader = "pytorch"

            self.model = registry.get(
                loader,
                model_path=model_path,
                wrapper_class=RealModelWrapper,
                learning_rate=0.001,
            )
            print(f"  Model loaded ({loader})")
            return True
        except Exception as e:
            print(f"  MODEL ERROR: {e}")
            import traceback

            traceback.print_exc()
            return False

    # ---------- P2P NETWORKING ----------

    def _process_message(self, msg: Dict):
        msg_type = msg.get("type", "")
        if msg_type == "model_update":
            with self._lock:
                sender = msg["client_id"]
                self.received_updates[sender] = {
                    "weights": msg["weights"],
                    "accuracy": msg.get("accuracy", 0.0),
                    "num_samples": msg.get("num_samples", 0),
                    "round": msg.get("round", 0),
                }
        elif msg_type == "shard_model":
            with self._lock:
                chief_id = msg["chief_id"]
                self.shard_models[chief_id] = {
                    "weights": msg["weights"],
                    "shard_id": msg.get("shard_id"),
                    "num_updates": msg.get("num_updates", 0),
                    "round": msg.get("round", 0),
                }
        elif msg_type == "global_model":
            with self._lock:
                self.global_weights = msg["weights"]
                self.round_num = msg.get("round", self.round_num)
                self._global_received.set()
        elif msg_type == "round_start":
            with self._lock:
                self.round_num = msg.get("round", self.round_num)
                self._round_started.set()

    def _make_handler(self):
        def handler(connected_node, data):
            try:
                if isinstance(data, dict):
                    self._process_message(data)
            except Exception as e:
                print(f"  MSG HANDLER ERROR: {e}")

        return handler

    def start_p2p(self) -> bool:
        try:
            from p2p_node import MyP2PNode

            self.node = MyP2PNode("0.0.0.0", P2P_PORT)
            self.node.max_connections = NUM_CLIENTS + 5
            self.node.node_message = self._make_handler()
            self.node.start()
            time.sleep(2)
            print(f"  P2P node started on port {P2P_PORT}")
            return True
        except Exception as e:
            print(f"  P2P ERROR: {e}")
            return False

    def connect_peers(self):
        print("  Connecting to peers...")
        for pid in range(NUM_CLIENTS):
            if pid == self.client_id:
                continue
            hostname = f"client-{pid}"
            for attempt in range(5):
                try:
                    self.node.connect_with_node(hostname, P2P_PORT)
                    print(f"    Connected to {hostname}")
                    break
                except Exception:
                    if attempt < 4:
                        time.sleep(1)
                    else:
                        print(f"    Could not connect to {hostname}")

    def broadcast(self, msg: Dict):
        if self.node:
            self.node.send_to_nodes(msg)

    def p2p_send(self, peer_id: int, msg: Dict):
        if peer_id == self.client_id:
            self._process_message(msg)
            return
        if not self.node:
            return
        hostname = f"client-{peer_id}"
        for peer in self.node.all_nodes:
            if peer.host == hostname or peer.id == f"client_{peer_id}":
                self.node.send_to_node(peer, msg)
                return
        self.broadcast(msg)

    # ---------- SERIALIZATION ----------

    def serialize(self, obj) -> str:
        return base64.b64encode(pickle.dumps(obj)).decode()

    @staticmethod
    def deserialize(data: str):
        return pickle.loads(base64.b64decode(data.encode()))

    # ---------- TRAINING ----------

    def local_training(self) -> Tuple[float, float]:
        for epoch in range(LOCAL_EPOCHS):
            indices = np.random.permutation(len(self.X_train))
            for i in range(0, len(self.X_train), BATCH_SIZE):
                batch_idx = indices[i : min(i + BATCH_SIZE, len(self.X_train))]
                X_batch = self.X_train[batch_idx]
                y_batch = self.y_train[batch_idx]
                self.model.train_step(X_batch, y_batch)
        loss = self.model.compute_loss(self.X_train, self.y_train)
        pred = self.model.predict(self.X_test)
        acc = float(np.mean(pred == self.y_test))
        return float(loss), acc

    def aggregate_fedavg(self, updates: List[Tuple[Dict, int]]) -> Dict:
        if not updates:
            return {}
        total = sum(c for _, c in updates)
        if total == 0:
            return updates[0][0]
        result = {}
        for weights, count in updates:
            w = count / total
            for key, val in weights.items():
                if key not in result:
                    result[key] = val * w
                else:
                    result[key] += val * w
        return result

    def average_dicts(self, dicts: List[Dict]) -> Dict:
        if not dicts:
            return {}
        result = {}
        for key in dicts[0]:
            result[key] = sum(d[key] for d in dicts) / len(dicts)
        return result

    def set_weights(self, weights: Dict):
        if self.model and weights:
            self.model.set_weights(weights)

    # ---------- WAIT HELPERS ----------

    def wait_for(
        self, container, key_type, expected_keys, timeout: float, label: str
    ) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                received = set(container.keys())
                expected = set(expected_keys)
                if expected.issubset(received):
                    return dict(container)
            time.sleep(2)
        with self._lock:
            result = dict(container)
        print(
            f"    TIMEOUT waiting for {label}: got {list(result.keys())}, expected {expected_keys}"
        )
        return result

    # ---------- ROUND PROTOCOL ----------

    def train_round(self, round_num: int) -> Dict:
        self.round_num = round_num
        self._global_received.clear()
        self._round_started.clear()

        print(f"\n{'=' * 60}")
        print(f"  ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'=' * 60}")

        # -- Phase 1: Local training --
        print(f"  [1/5] Local training...")
        train_loss, test_acc = self.local_training()
        print(f"    Loss={train_loss:.4f}, Test Acc={test_acc:.4f}")

        weights = self.model.get_weights()
        metrics = {
            "round": round_num,
            "client": self.client_name,
            "shard": self.shard_id,
            "train_loss": train_loss,
            "test_accuracy": test_acc,
            "num_train_samples": self.num_train_samples,
            "num_test_samples": self.num_test_samples,
        }

        weights_b64 = self.serialize(weights)

        # -- Phase 2: Store own update locally --
        self._process_message(
            {
                "type": "model_update",
                "client_id": self.client_name,
                "weights": weights_b64,
                "accuracy": test_acc,
                "num_samples": self.num_train_samples,
                "round": round_num,
            }
        )

        # Also send to chief via P2P (if we're not the chief)
        if not self.is_chief:
            print(f"  [2/5] Sending to chief client_{self.shard_chief_id}...")
            self.p2p_send(
                self.shard_chief_id,
                {
                    "type": "model_update",
                    "client_id": self.client_name,
                    "weights": weights_b64,
                    "accuracy": test_acc,
                    "num_samples": self.num_train_samples,
                    "round": round_num,
                },
            )
        else:
            print(f"  [2/5] I am the chief, own update stored locally")

        # -- Phase 3: Shard chief aggregation --
        if self.is_chief:
            print(f"  [3/5] Chief collecting shard updates...")
            other_members = [
                f"client_{c}" for c in self.shard_members if c != self.client_id
            ]
            self.wait_for(
                self.received_updates,
                str,
                other_members,
                timeout=90,
                label="shard member updates",
            )

            with self._lock:
                all_updates = dict(self.received_updates)

            updates_list = []
            for cid, upd in all_updates.items():
                w = self.deserialize(upd["weights"])
                n = upd.get("num_samples", 1)
                updates_list.append((w, n))

            shard_aggregated = self.aggregate_fedavg(updates_list)
            shard_b64 = self.serialize(shard_aggregated)
            num_collected = len(all_updates)
            print(f"    Aggregated {num_collected} updates")

            shard_payload = {
                "type": "shard_model",
                "chief_id": self.client_id,
                "shard_id": self.shard_id,
                "weights": shard_b64,
                "num_updates": num_collected,
                "round": round_num,
            }

            if self.is_coordinator:
                print(f"  [3/5] Coordinator-chief: storing shard aggregate locally")
                self._process_message(shard_payload)
            else:
                print(f"  [3/5] Sending shard aggregate to coordinator")
                self.p2p_send(0, shard_payload)

            metrics["shard_updates"] = num_collected
        else:
            print(f"  [3/5] Waiting for chief aggregation...")

        # -- Phase 4: Global aggregation --
        if self.is_coordinator:
            print(f"  [4/5] Coordinator collecting shard models...")
            other_chiefs = []
            for s_id in range(NUM_SHARDS):
                s_start = s_id * clients_per_shard
                if s_start != self.client_id:
                    other_chiefs.append(s_start)

            shard_results = self.wait_for(
                self.shard_models,
                int,
                other_chiefs,
                timeout=120,
                label="shard chief models",
            )

            with self._lock:
                all_shard_models = dict(self.shard_models)

            if all_shard_models:
                shard_weights = [
                    self.deserialize(r["weights"]) for r in all_shard_models.values()
                ]
                global_weights = self.average_dicts(shard_weights)
                with self._lock:
                    self.global_weights = global_weights

                global_b64 = self.serialize(global_weights)
                self.broadcast(
                    {
                        "type": "global_model",
                        "weights": global_b64,
                        "round": round_num,
                    }
                )
                self.set_weights(global_weights)
                print(
                    f"    Global model aggregated from {len(all_shard_models)} shards and broadcast"
                )
                metrics["shards_collected"] = len(all_shard_models)
            else:
                print(f"    WARNING: No shard models collected")
                metrics["shards_collected"] = 0
        else:
            print(f"  [4/5] Waiting for global model...")
            received = self._global_received.wait(timeout=120)
            if received:
                with self._lock:
                    if self.global_weights is not None:
                        self.set_weights(self.deserialize(self.global_weights))
                print(f"    Global model received")
            else:
                print(f"    WARNING: Global model timeout")

        # -- Phase 5: Logging --
        with self._lock:
            model_updated = self.global_weights is not None
        metrics["model_updated"] = model_updated
        self.round_metrics.append(metrics)

        # Clear per-round state
        self.received_updates.clear()
        self.shard_models.clear()
        self._global_received.clear()

        return metrics

    # ---------- RESULTS ----------

    def save_results(self):
        results_root = Path(RESULTS_DIR)
        if RESULTS_SUBDIR:
            results_root = results_root / RESULTS_SUBDIR
        results_path = results_root / f"results_client_{self.client_id}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "client_id": self.client_name,
            "shard_id": self.shard_id,
            "is_chief": self.is_chief,
            "is_coordinator": self.is_coordinator,
            "config": {
                "num_clients": NUM_CLIENTS,
                "num_shards": NUM_SHARDS,
                "num_rounds": NUM_ROUNDS,
                "local_epochs": LOCAL_EPOCHS,
            },
            "final_test_accuracy": self.round_metrics[-1]["test_accuracy"]
            if self.round_metrics
            else 0,
            "rounds": self.round_metrics,
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {results_path}")
        print(
            f"  RESULTS JSON: {json.dumps({'client': self.client_name, 'final_acc': results['final_test_accuracy'], 'rounds': len(results['rounds'])})}"
        )

    # ---------- MAIN ----------

    def run(self):
        try:
            if RUNNER_MODE == "contribution":
                from federated_learning_system import ShardedFederatedLearningSystem

                print("Running contribution mode...")
                system = ShardedFederatedLearningSystem(
                    num_clients=NUM_CLIENTS,
                    num_shards=NUM_SHARDS,
                    num_rounds=NUM_ROUNDS,
                    pretrained_model_path="global_model.npy",
                    use_real_data=True,
                    real_data_dir=DATA_DIR,
                )

                output_root = Path(RESULTS_DIR) / "contribution_results"
                if RUN_LABEL:
                    output_root = output_root / RUN_LABEL
                elif RESULTS_SUBDIR:
                    output_root = output_root / RESULTS_SUBDIR
                output_root.mkdir(parents=True, exist_ok=True)
                output_file = output_root / "training_results.json"

                system.execute_full_training()
                system.save_results(str(output_file))
                print(f"Contribution results saved to {output_file}")
                return

            print("Initializing...")
            sys.stdout.flush()

            if not self.load_data():
                print("FATAL: data loading failed")
                sys.stdout.flush()
                return
            if not self.init_model():
                print("FATAL: model init failed")
                sys.stdout.flush()
                return
            if not self.start_p2p():
                print("FATAL: P2P start failed")
                sys.stdout.flush()
                return

            self.connect_peers()
            time.sleep(5)

            with open("/tmp/client_ready", "w") as f:
                f.write("ready")

            connected = len(self.node.all_nodes) if self.node else 0
            print(f"Ready: connected to {connected} peers")
            print(f"Starting {NUM_ROUNDS} rounds...")
            sys.stdout.flush()

            for r in range(1, NUM_ROUNDS + 1):
                self.train_round(r)
                time.sleep(2)
                sys.stdout.flush()

            self.save_results()
            avg_acc = float(np.mean([m["test_accuracy"] for m in self.round_metrics]))
            final_acc = self.round_metrics[-1]["test_accuracy"]
            print(f"\n  {'=' * 50}")
            print(f"  TRAINING COMPLETE")
            print(f"  Avg Test Accuracy:  {avg_acc:.4f}")
            print(f"  Final Test Accuracy: {final_acc:.4f}")
            print(f"  {'=' * 50}")
            sys.stdout.flush()
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            import traceback

            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    client = DockerFLClient()
    client.run()
