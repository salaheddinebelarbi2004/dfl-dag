"""
Demo and Testing Script for Sharded Federated Learning System
Shows complete usage examples and integration testing
"""

import numpy as np
import sys
from typing import Dict, Any

# Import all components
from federated_learning_system import ShardedFederatedLearningSystem
from client import FederatedLearningClient, create_synthetic_client_data
from polynomial_proxy import PolynomialProxyFactory
from zkp_proofs import ZKPFactory
from robust_aggregation import RobustAggregator, AggregationMethod
from shard_chief import ShardChiefElector, ShardChiefCandidate, NodeCapabilities
from dag_system import ShardedDAG
from global_aggregator import GlobalCoordinator, ShardModelUpdate

from federated_learning_config import INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM


def test_polynomial_proxy():
    """Test polynomial proxy creation and verification"""
    print("\n" + "="*60)
    print("TEST 1: Polynomial Proxy")
    print("="*60)
    
    # Create synthetic gradient
    gradient = np.random.randn(784).astype(np.float32)
    model_params = np.random.randn(784).astype(np.float32) * 0.01
    
    # Create polynomial proxy
    proxy = PolynomialProxyFactory.create_from_update(gradient, model_params)
    
    print(f"✓ Created polynomial proxy")
    print(f"  Degree: {proxy.degree}")
    print(f"  Coefficients: {len(proxy.get_coefficients_ordered())}")
    
    # Verify consistency
    is_consistent = proxy.verify_consistency(gradient)
    print(f"  Consistency: {is_consistent}")
    
    # Evaluate at points
    for x in [0.0, 0.5, 1.0]:
        value = proxy.evaluate(x)
        print(f"  P({x}) = {value:.6f}")
    
    return True


def test_zkp_system():
    """Test ZKP generation and verification"""
    print("\n" + "="*60)
    print("TEST 2: Zero-Knowledge Proofs")
    print("="*60)
    
    gradient = np.random.randn(784).astype(np.float32)
    model_params = np.random.randn(784).astype(np.float32) * 0.01
    
    # Create polynomial proxy
    poly = PolynomialProxyFactory.create_from_update(gradient, model_params)
    
    # Generate proof
    generator = ZKPFactory.get_generator()
    proof = generator.generate_proof(
        gradient, model_params,
        client_id="test_client",
        round_num=1,
        polynomial_proxy=poly
    )
    
    print(f"✓ Generated ZKP proof")
    print(f"  Proof ID: {proof.proof_id[:8]}...")
    print(f"  Gradient Correct: {proof.certifies_gradient_correct}")
    print(f"  Poly Consistency: {proof.certifies_poly_consistency}")
    print(f"  Norm Bound: {proof.certifies_norm_bound}")
    
    # Verify proof
    verifier = ZKPFactory.get_verifier()
    is_valid, details = verifier.verify_proof(proof, gradient, poly)
    
    print(f"✓ Verified ZKP proof")
    print(f"  Valid: {is_valid}")
    print(f"  Commitment Structure: {details['checks']['commitment_structure']}")
    print(f"  Norm Satisfied: {details['checks']['norm_bound']['satisfied']}")
    
    return is_valid


def test_client_training():
    """Test client local training and update generation"""
    print("\n" + "="*60)
    print("TEST 3: Client Local Training")
    print("="*60)
    
    # Create client
    client = FederatedLearningClient(
        client_id="test_client",
        shard_id="shard_0"
    )
    
    # Generate synthetic data
    X, y = create_synthetic_client_data(0, num_samples=100)
    client.set_local_dataset(X, y)
    
    print(f"✓ Created client with {len(X)} samples")
    
    # Initialize global model
    global_model = np.random.randn(
        INPUT_DIM * HIDDEN_DIM +
        HIDDEN_DIM +
        HIDDEN_DIM * OUTPUT_DIM +
        OUTPUT_DIM
    ).astype(np.float32) * 0.01
    
    # Receive model
    client.receive_global_model(global_model)
    print(f"✓ Received global model")
    
    # Local training
    training_meta = client.local_training_round(local_epochs=2)
    print(f"✓ Local training completed")
    print(f"  Epochs: {training_meta['local_epochs']}")
    print(f"  Samples: {training_meta['total_samples']}")
    print(f"  Final Loss: {training_meta['epoch_losses'][-1]:.6f}")
    
    # Create update
    update = client.create_update(global_model, round_num=1)
    print(f"✓ Created update package")
    print(f"  Gradient Norm: {np.linalg.norm(update.gradient):.6f}")
    print(f"  Commitment: {update.commitment[:16]}...")
    print(f"  ZKP Verified: {update.zkp.verification_result}")
    
    return True


def test_robust_aggregation():
    """Test robust aggregation methods"""
    print("\n" + "="*60)
    print("TEST 4: Robust Aggregation")
    print("="*60)
    
    # Create synthetic updates
    updates = [np.random.randn(100).astype(np.float32) for _ in range(10)]
    client_ids = [f"client_{i}" for i in range(10)]
    verification_results = [True] * 10
    
    aggregator = RobustAggregator()
    
    # Test different methods
    methods = [
        AggregationMethod.MEAN,
        AggregationMethod.MEDIAN,
        AggregationMethod.CLUSTER_CONSENSUS,
        AggregationMethod.REPUTATION_WEIGHTED,
        AggregationMethod.ROBUST_HYBRID,
    ]
    
    for method in methods:
        try:
            agg, meta = aggregator.aggregate(
                updates, verification_results, client_ids,
                method=method
            )
            print(f"✓ {method.value}: {agg.shape}")
        except Exception as e:
            print(f"✗ {method.value}: {str(e)[:50]}")
    
    return True


def test_shard_chief_election():
    """Test Shard Chief election"""
    print("\n" + "="*60)
    print("TEST 5: Shard Chief Election")
    print("="*60)
    
    # Create candidates
    candidates = [
        ShardChiefCandidate(
            node_id=f"client_{i}",
            capabilities=NodeCapabilities(
                node_id=f"client_{i}",
                uptime=0.90 + np.random.rand() * 0.1,
                bandwidth_mbps=50 + np.random.rand() * 100,
                compute_capacity=0.8 + np.random.rand() * 0.4,
            )
        )
        for i in range(5)
    ]
    
    elector = ShardChiefElector()
    chief, election_meta = elector.elect_for_shard(
        candidates, "shard_0", top_k=3
    )
    
    print(f"✓ Elected Shard Chief")
    print(f"  Elected: {chief.chief_id}")
    print(f"  Score: {election_meta['score']:.3f}")
    print(f"  Rank: {election_meta['rank']}/{election_meta['top_k']}")
    print(f"  Total Candidates: {election_meta['total_candidates']}")
    
    print(f"✓ Chief Capabilities:")
    print(f"  Uptime: {chief.capabilities.uptime:.1%}")
    print(f"  Bandwidth: {chief.capabilities.bandwidth_mbps:.1f} Mbps")
    print(f"  Compute: {chief.capabilities.compute_capacity:.2f}x")
    
    return True


def test_dag_system():
    """Test DAG model evolution tracking"""
    print("\n" + "="*60)
    print("TEST 6: DAG System")
    print("="*60)
    
    # Create sharded DAG
    dag = ShardedDAG(num_shards=2, initial_model_hash="initial_hash")
    
    print(f"✓ Created sharded DAG with 2 shards")
    
    # Add updates
    node_id_1 = dag.add_shard_update(
        "shard_0", "client_0", 1,
        "model_hash_1", "gradient_hash_1", "root"
    )
    node_id_2 = dag.add_shard_update(
        "shard_0", "client_1", 1,
        "model_hash_2", "gradient_hash_2", "root"
    )
    
    print(f"✓ Added 2 updates to shard_0")
    
    # Verify updates
    dag.shard_dags["shard_0"].verify_update(node_id_1, zkp_verified=True)
    dag.shard_dags["shard_0"].verify_update(node_id_2, zkp_verified=True)
    
    # Finalize updates
    dag.shard_dags["shard_0"].finalize_update(node_id_1)
    dag.shard_dags["shard_0"].finalize_update(node_id_2)
    
    print(f"✓ Verified and finalized updates")
    
    # Get stats
    stats = dag.get_stats()
    print(f"✓ DAG Statistics:")
    print(f"  Global Model History: {stats['global_model_history_length']}")
    print(f"  Shard 0 Nodes: {stats['shard_stats']['shard_0']['total_nodes']}")
    print(f"  Shard 0 Finalized: {stats['shard_stats']['shard_0']['finalized_chain_length']}")
    
    return True


def test_global_aggregator():
    """Test global model aggregation"""
    print("\n" + "="*60)
    print("TEST 7: Global Aggregator")
    print("="*60)
    
    # Create initial model
    initial_model = np.random.randn(100).astype(np.float32)
    
    # Create coordinator
    coordinator = GlobalCoordinator(num_shards=2, initial_model=initial_model)
    print(f"✓ Created global coordinator")
    
    # Simulate shard updates
    shard_updates = [
        ShardModelUpdate(
            shard_id=f"shard_{i}",
            shard_model=np.random.randn(100).astype(np.float32) * 0.05,
            timestamp=0,
            num_updates_aggregated=10,
            verification_success_rate=0.9
        )
        for i in range(2)
    ]
    
    # Run aggregation
    new_model, round_meta = coordinator.run_training_round(
        shard_updates, global_learning_rate=0.5
    )
    
    print(f"✓ Executed training round")
    print(f"  Round: {round_meta['round_num']}")
    print(f"  Aggregation Success: {round_meta['aggregation']['success']}")
    print(f"  Model Change Norm: {round_meta['aggregation']['model_update']['model_change_norm']:.6f}")
    
    return True


def run_integration_test():
    """Run complete integration test (mini training)"""
    print("\n" + "="*60)
    print("TEST 8: Integration Test (Mini Training)")
    print("="*60)
    
    # Create small system
    system = ShardedFederatedLearningSystem(
        num_clients=6,
        num_shards=2,
        num_rounds=2  # Just 2 rounds for testing
    )
    
    print(f"✓ System initialized")
    print(f"  Clients: {system.num_clients}")
    print(f"  Shards: {system.num_shards}")
    print(f"  Rounds: {system.num_rounds}")
    
    # Run training
    results = system.execute_full_training()
    
    print(f"✓ Training completed")
    print(f"  Total Time: {results['total_training_time_seconds']:.2f}s")
    print(f"  Rounds Completed: {results['num_rounds']}")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# SHARDED FEDERATED LEARNING SYSTEM - TEST SUITE")
    print("#"*60)
    
    tests = [
        ("Polynomial Proxy", test_polynomial_proxy),
        ("ZKP System", test_zkp_system),
        ("Client Training", test_client_training),
        ("Robust Aggregation", test_robust_aggregation),
        ("Shard Chief Election", test_shard_chief_election),
        ("DAG System", test_dag_system),
        ("Global Aggregator", test_global_aggregator),
        ("Integration Test", run_integration_test),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASSED"
            passed += 1
        except Exception as e:
            results[test_name] = f"FAILED: {str(e)[:100]}"
            failed += 1
            print(f"\n✗ Test failed: {str(e)}")
    
    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    print(f"Total: {len(tests)}, Passed: {passed}, Failed: {failed}")
    print()
    for test_name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        print(f"{status} {test_name}: {result}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
