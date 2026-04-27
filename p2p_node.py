import time
import json
from p2pnetwork.node import Node
import threading
import random
import uuid
from typing import Dict, Any, Optional
import pickle
import base64
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# =============== FLOWR CORE ===============
# Pure orchestration logic - NO networking
class FlowrCore:
    """
    Flowr Core: Task orchestration engine
    - Creates and manages tasks
    - Tracks task status and results
    - Pure logic only (no network calls)
    """
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> task info
        self.results: Dict[str, Any] = {}  # task_id -> result
        self.adapter: Optional['FlowrAdapter'] = None

    def set_adapter(self, adapter: 'FlowrAdapter'):
        """Register the adapter (bridge to P2P network)"""
        self.adapter = adapter

    def create_task(self, action: str, payload: Any) -> str:
        """Create a task and return task_id"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "id": task_id,
            "action": action,
            "payload": payload,
            "status": "pending",
            "created_at": time.time()
        }
        return task_id

    def send_task(self, action: str, payload: Any) -> str:
        """Create task and send it via adapter"""
        task_id = self.create_task(action, payload)

        if self.adapter:
            self.adapter.send_task(task_id, action, payload)
        else:
            print("⚠️  No adapter registered - task not sent")

        return task_id

    def mark_running(self, task_id: str):
        """Mark task as running"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "running"

    def mark_completed(self, task_id: str, result: Any):
        """Mark task as completed and store result"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            self.results[task_id] = result
            print(f"✅ Task {task_id} completed: {result}")

    def get_task_status(self, task_id: str) -> str:
        """Get task status"""
        if task_id in self.tasks:
            return self.tasks[task_id]["status"]
        return "unknown"

    def get_result(self, task_id: str) -> Any:
        """Get task result"""
        return self.results.get(task_id)


# =============== FLOWR ADAPTER ===============
# Bridge between FlowrCore and P2P network
class FlowrAdapter:
    """
    Flowr Adapter: Connects FlowrCore to P2P network
    - Sends Flowr tasks through P2P messages
    - Receives P2P messages and converts to Flowr events
    - Maintains message format contract
    """
    def __init__(self, flowr_core: FlowrCore):
        self.flowr_core = flowr_core
        self.node: Optional[MyP2PNode] = None
        self.peer_list = []  # List of available peers

    def set_node(self, node: 'MyP2PNode'):
        """Register the P2P node"""
        self.node = node

    def update_peers(self, peers: list):
        """Update available peer list (for peer selection)"""
        self.peer_list = peers

    def select_peer(self) -> Optional[Any]:
        """Select a peer to send task to (random for now)"""
        if self.node and self.node.all_nodes:
            return random.choice(self.node.all_nodes)
        return None

    def send_task(self, task_id: str, action: str, payload: Any):
        """Send Flowr task through P2P network"""
        message = {
            "type": "task",
            "task_id": task_id,
            "action": action,
            "payload": payload
        }

        peer = self.select_peer()
        if peer:
            self.node.send_to_node(peer, message)
            self.flowr_core.mark_running(task_id)
            print(f"📤 Flowr Task {task_id} sent to {peer.id}")
        else:
            print(f"⚠️  No peers available to send task {task_id}")

    def handle_result(self, task_id: str, result: Any):
        """Handle result coming back from P2P network"""
        self.flowr_core.mark_completed(task_id, result)


class MyP2PNode(Node):
    def __init__(self, host, port, flowr_adapter: Optional[FlowrAdapter] = None):
        super().__init__(host, port, max_connections=10)
        self.flowr_adapter = flowr_adapter

    def outbound_node_connected(self, connected_node):
        print(f"✅ OUTBOUND connected to {connected_node.id} ({connected_node.host}:{connected_node.port})")
        if self.flowr_adapter:
            self.flowr_adapter.update_peers(self.all_nodes)

    def inbound_node_connected(self, connected_node):
        print(f"✅ INBOUND connection from {connected_node.id} ({connected_node.host}:{connected_node.port})")
        if self.flowr_adapter:
            self.flowr_adapter.update_peers(self.all_nodes)

    def node_message(self, connected_node, data):
        """
        Route messages based on type:
        - "task" → execute task, send result
        - "result" → forward to FlowrAdapter
        - legacy → treat as regular message
        """
        # Handle Flowr task messages
        if isinstance(data, dict) and data.get("type") == "task":
            task_id = data.get("task_id")
            action = data.get("action")
            payload = data.get("payload")
            print(f"📨 Flowr Task received from {connected_node.id}: {action}")

            result = self.handle_task(action, payload)

            # Send result back using Flowr message format
            response = {
                "type": "result",
                "task_id": task_id,
                "result": result
            }
            self.send_to_node(connected_node, response)
            print(f"📤 Result sent back to {connected_node.id}: {result}")

        # Handle Flowr result messages
        elif isinstance(data, dict) and data.get("type") == "result":
            task_id = data.get("task_id")
            result = data.get("result")
            print(f"📨 Flowr Result received from {connected_node.id}: {result}")

            if self.flowr_adapter:
                self.flowr_adapter.handle_result(task_id, result)

        # Legacy message handling (backward compatibility)
        else:
            print(f"📨 Message from {connected_node.id}: {data}")

    def handle_task(self, action: str, payload: Any) -> Any:
        """
        Execute task based on action type
        Supported actions: "sum", "ping", "train_tree", "predict_tree"
        """
        try:
            if action == "sum":
                # Sum a list of numbers
                if isinstance(payload, list):
                    result = sum(payload)
                    return {"status": "success", "value": result}
                else:
                    return {"status": "error", "message": "Payload must be a list"}

            elif action == "ping":
                # Simple ping-pong
                return {"status": "success", "message": "pong"}

            elif action == "train_tree":
                # Train Decision Tree Classifier
                # payload format: {"X": [[features...], ...], "y": [labels...], "max_depth": 5}
                return self._train_decision_tree(payload)

            elif action == "predict_tree":
                # Make predictions with tree
                # payload format: {"model": encoded_model, "X": [[features...], ...]}
                return self._predict_with_tree(payload)

            else:
                return {"status": "error", "message": f"Unknown task: {action}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _train_decision_tree(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a Decision Tree Classifier on provided data

        Payload:
        {
            "X": [[feature1, feature2, ...], ...],  # Training features
            "y": [label1, label2, ...],              # Training labels
            "max_depth": 5,                          # Optional: max tree depth
            "test_X": [...],                         # Optional: test data
            "test_y": [...]                          # Optional: test labels
        }
        """
        try:
            X = np.array(payload.get("X"))
            y = np.array(payload.get("y"))
            max_depth = payload.get("max_depth", 5)

            if X.size == 0 or y.size == 0:
                return {"status": "error", "message": "X and y cannot be empty"}

            # Train decision tree
            tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            tree.fit(X, y)

            # Get training accuracy
            train_pred = tree.predict(X)
            train_accuracy = accuracy_score(y, train_pred)

            # Get test accuracy if provided
            test_accuracy = None
            if "test_X" in payload and "test_y" in payload:
                test_X = np.array(payload.get("test_X"))
                test_y = np.array(payload.get("test_y"))
                test_pred = tree.predict(test_X)
                test_accuracy = accuracy_score(test_y, test_pred)

            # Serialize model to base64
            model_bytes = pickle.dumps(tree)
            model_encoded = base64.b64encode(model_bytes).decode('utf-8')

            # Get feature importances
            feature_importance = tree.feature_importances_.tolist()

            result = {
                "status": "success",
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy) if test_accuracy else None,
                "tree_depth": int(tree.get_depth()),
                "n_nodes": tree.tree_.node_count,
                "feature_importance": feature_importance,
                "model": model_encoded,  # Encoded model for distributed learning
                "n_samples": len(X),
                "n_features": X.shape[1]
            }
            return result

        except Exception as e:
            return {"status": "error", "message": f"Training failed: {str(e)}"}

    def _predict_with_tree(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions using a trained Decision Tree model

        Payload:
        {
            "model": encoded_model,           # Base64 encoded model
            "X": [[feature1, feature2, ...]]  # Data to predict
        }
        """
        try:
            model_encoded = payload.get("model")
            X = np.array(payload.get("X"))

            # Deserialize model
            model_bytes = base64.b64decode(model_encoded.encode('utf-8'))
            tree = pickle.loads(model_bytes)

            # Make predictions
            predictions = tree.predict(X)
            probabilities = tree.predict_proba(X)

            result = {
                "status": "success",
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "n_predictions": len(predictions)
            }
            return result

        except Exception as e:
            return {"status": "error", "message": f"Prediction failed: {str(e)}"}

    def node_disconnect(self, connected_node):
        print(f"❌ Disconnected from {connected_node.id}")
        if self.flowr_adapter:
            self.flowr_adapter.update_peers(self.all_nodes)


# =============== START THE NODE ===============
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python p2p_node.py <host> <port>")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    # Initialize Flowr Core and Adapter
    flowr_core = FlowrCore()
    flowr_adapter = FlowrAdapter(flowr_core)

    # Create P2P node with Flowr integration
    node = MyP2PNode(host, port, flowr_adapter)
    flowr_adapter.set_node(node)
    flowr_core.set_adapter(flowr_adapter)

    node.start()
    print(f"🚀 Node started on {host}:{port}")
    print(f"   Your Node ID: {node.id}")
    print(f"\n⚙️  Flowr Core initialized")
    print("\nCommands:")
    print("   send <message>              → Send message to all connected nodes")
    print("   connect <host> <port>       → Connect to another node")
    print("   task <action> <payload>     → Send Flowr task (e.g., 'task sum [1,2,3]')")
    print("   status <task_id>            → Check task status")
    print("   result <task_id>            → Get task result")
    print("   peers                       → Show connected peers")
    print("   quit                        → Stop the node\n")

    # Run the menu in the main thread
    try:
        while True:
            cmd = input(">> ").strip()

            if cmd.lower() == "quit" or cmd.lower() == "exit":
                break

            elif cmd.startswith("send "):
                message = cmd[5:].strip()
                try:
                    # Try to send as JSON dict if possible
                    data = json.loads(message)
                except:
                    data = {"message": message, "from": node.id}
                node.send_to_nodes(data)
                print(f"📤 Sent to all nodes: {data}")

            elif cmd.startswith("task "):
                # Parse: task <action> <payload>
                parts = cmd[5:].strip().split(" ", 1)
                if len(parts) >= 1:
                    action = parts[0]
                    try:
                        payload = json.loads(parts[1]) if len(parts) > 1 else None
                    except:
                        payload = parts[1] if len(parts) > 1 else None

                    task_id = flowr_core.send_task(action, payload)
                    print(f"📋 Task created: {task_id}")
                else:
                    print("Usage: task <action> <payload>")

            elif cmd.startswith("status "):
                task_id = cmd[7:].strip()
                status = flowr_core.get_task_status(task_id)
                print(f"📊 Task {task_id}: {status}")

            elif cmd.startswith("result "):
                task_id = cmd[7:].strip()
                result = flowr_core.get_result(task_id)
                if result:
                    print(f"✅ Result: {result}")
                else:
                    print(f"⚠️  No result found for task {task_id}")

            elif cmd.startswith("connect "):
                parts = cmd.split()
                if len(parts) == 3:
                    node.connect_with_node(parts[1], int(parts[2]))
                else:
                    print("Usage: connect <host> <port>")

            elif cmd == "peers":
                print("Connected nodes:")
                for n in node.all_nodes:
                    print(f"   {n.id}  →  {n.host}:{n.port}")

            else:
                print("Unknown command. Type 'help' or see available commands above")

    except KeyboardInterrupt:
        pass
    finally:
        print("\n🛑 Stopping node...")
        node.stop()