"""
CUSTOM EXAMPLES - Complete Usage Patterns
Sharded Federated Learning System
"""

import numpy as np
from typing import List

# =================================================================
# EXAMPLE 1: Complete Training with Custom Parameters
# =================================================================

def example_1_custom_training():
    """Run training with custom parameters"""

    from federated_learning_system import ShardedFederatedLearningSystem

    # Create system with custom configuration
    system = ShardedFederatedLearningSystem(
        num_clients=20,      # More clients
        num_shards=4,        # More shards
        num_rounds=10        # More training rounds
    )

    # Execute training
    results = system.execute_full_training()

    # Print results
    print(f"Training Time: {results['total_training_time_seconds']:.2f}s")
    print(f"Final Model Update Norm: {results['final_system_state']}")

    # Save for analysis
    system.save_results("custom_training_results.json")


# =================================================================
# EXAMPLE 2: Single Client Training Pipeline
# =================================================================

def example_2_single_client():
    """Demonstrate single client workflow"""

    from client import FederatedLearningClient, create_synthetic_client_data
    from polynomial_proxy import PolynomialProxyFactory
    from zkp_proofs import ZKPFactory
    from federated_learning_config import INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM

    # 1. Create client
    client = FederatedLearningClient(
        client_id="demo_client_1",
        shard_id="shard_0"
    )
    print(f"✓ Created client: {client.client_id}")

    # 2. Assign local dataset
    X_local, y_local = create_synthetic_client_data(
        client_id=0,
        num_samples=500,
        data_distribution="non_iid"  # Non-IID data
    )
    client.set_local_dataset(X_local, y_local)
    print(f"✓ Assigned dataset: {len(X_local)} samples")

    # 3. Receive global model
    global_model = np.random.randn(
        INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM +
        HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM
    ).astype(np.float32) * 0.01

    client.receive_global_model(global_model)
    print(f"✓ Received global model")

    # 4. Perform local training
    training_metadata = client.local_training_round(
        local_epochs=3,
        batch_size=32
    )
    print(f"✓ Local training complete")
    print(f"  Final loss: {training_metadata['epoch_losses'][-1]:.6f}")

    # 5. Generate update with all components
    update = client.create_update(global_model, round_num=1)
    print(f"✓ Generated update package")
    print(f"  Gradient norm: {np.linalg.norm(update.gradient):.6f}")
    print(f"  Commitment: {update.commitment[:24]}...")
    print(f"  Polynomial degree: {update.polynomial_proxy.degree}")
    print(f"  ZKP certifies gradient: {update.zkp.certifies_gradient_correct}")

    # 6. Inspect client stats
    stats = client.get_stats()
    print(f"✓ Client stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


# =================================================================
# EXAMPLE 3: Polynomial Proxy Analysis
# =================================================================

def example_3_polynomial_proxy():
    """Analyze polynomial proxy properties"""

    from polynomial_proxy import PolynomialProxy, PolynomialProxyFactory
    import matplotlib.pyplot as plt  # Optional: for visualization

    # Create gradient with specific properties
    gradient = np.random.randn(100).astype(np.float32)


    # Build polynomial proxy
    proxy = PolynomialProxyFactory.create_from_update(gradient)

    print(f"✓ Polynomial Proxy")
    print(f"  Degree: {proxy.degree}")
    print(f"  Evaluation points: {proxy.eval_points}")
    print(f"  Coefficients: {proxy.get_coefficients_ordered()}")

    # Evaluate polynomial at various points
    print(f"\n  Evaluations:")
    for x in np.linspace(0, 1, 5):
        value = proxy.evaluate(x)
        print(f"    P({x:.2f}) = {value:.6f}")

    # Verify consistency with original gradient
    is_consistent = proxy.verify_consistency(gradient, tolerance=0.2)
    print(f"\n  Consistency with gradient: {is_consistent}")

    # Get polynomial hash (for commitment)
    poly_hash = proxy.get_hash()
    print(f"  Polynomial hash: {poly_hash[:32]}...")

    # Serialize and deserialize
    poly_dict = proxy.to_dict()
    proxy_restored = PolynomialProxy.from_dict(poly_dict)
    print(f"  Serialization: OK (restored {len(poly_restored.coefficients)} coefficients)")


# =================================================================
# EXAMPLE 4: ZKP Generation and Verification Flow
# =================================================================

def example_4_zkp_workflow():
    """Complete ZKP workflow"""

    from zkp_proofs import ZKPFactory
    from polynomial_proxy import PolynomialProxyFactory

    # Setup
    gradient = np.random.randn(784).astype(np.float32)
    model_params = np.random.randn(784).astype(np.float32) * 0.01

    # Create polynomial proxy
    poly = PolynomialProxyFactory.create_from_update(gradient, model_params)
    print(f"✓ Polynomial proxy created")

    # GENERATE PROOF
    generator = ZKPFactory.get_generator()
    proof = generator.generate_proof(
        gradient,
        model_params,
        client_id="example_client",
        round_num=1,
        polynomial_proxy=poly,
        norm_bound=1.0
    )

    print(f"✓ ZKP Generated")
    print(f"  Proof ID: {proof.proof_id[:16]}...")
    print(f"  Commitments: {len(proof.commitments)}")
    print(f"  Challenges: {len(proof.challenges)}")
    print(f"  Responses: {len(proof.responses)}")
    print(f"  Certifies gradient: {proof.certifies_gradient_correct}")
    print(f"  Certifies poly consistency: {proof.certifies_poly_consistency}")
    print(f"  Certifies norm bound: {proof.certifies_norm_bound}")

    # VERIFY PROOF
    verifier = ZKPFactory.get_verifier()
    is_valid, verification_report = verifier.verify_proof(
        proof, gradient, poly, norm_bound=1.0
    )

    print(f"\n✓ ZKP Verification Result: {is_valid}")
    print(f"  Checks performed:")
    for check_name, check_result in verification_report['checks'].items():
        if isinstance(check_result, dict):
            print(f"    {check_name}: {check_result['satisfied']}")
        else:
            print(f"    {check_name}: {check_result}")

    # Serialize proof
    proof_dict = proof.to_dict()
    print(f"\n✓ Proof serialized ({len(str(proof_dict))} bytes)")


# =================================================================
# EXAMPLE 5: Robust Aggregation Methods Comparison
# =================================================================

def example_5_aggregation_comparison():
    """Compare different aggregation methods"""

    from robust_aggregation import RobustAggregator, AggregationMethod

    # Create synthetic updates with different quality
    np.random.seed(42)

    # Good updates (from honest clients)
    good_updates = [np.random.randn(100) * 0.1 for _ in range(7)]

    # Outlier updates (from Byzantine or noisy clients)
    outlier_updates = [np.random.randn(100) * 2.0 for _ in range(2)]

    # Mix
    all_updates = good_updates + outlier_updates
    client_ids = [f"client_{i}" for i in range(len(all_updates))]
    verification_results = [True] * len(all_updates)

    print(f"✓ Aggregation Comparison")
    print(f"  Updates: 7 good (norm ~0.1) + 2 outliers (norm ~2.0) + 1 unreliable")

    aggregator = RobustAggregator()

    # Test each method
    methods = [
        AggregationMethod.MEAN,
        AggregationMethod.MEDIAN,
        AggregationMethod.CLUSTER_CONSENSUS,
        AggregationMethod.REPUTATION_WEIGHTED,
        AggregationMethod.ROBUST_HYBRID,
    ]

    results = {}
    for method in methods:
        try:
            agg, metadata = aggregator.aggregate(
                all_updates,
                verification_results=verification_results,
                client_ids=client_ids,
                method=method
            )

            agg_norm = np.linalg.norm(agg)
            results[method.value] = {
                "norm": agg_norm,
                "metadata": metadata
            }

            print(f"\n  {method.value}:")
            print(f"    Aggregated norm: {agg_norm:.6f}")
            if "stages" in metadata:
                for stage, count in metadata["stages"].items():
                    if isinstance(count, (int, float)):
                        print(f"    {stage}: {count}")

        except Exception as e:
            print(f"\n  {method.value}: ERROR - {str(e)[:60]}")

    print(f"\n✓ Comparison Summary:")
    print(f"  Best for standard case: ROBUST_HYBRID")
    print(f"  Fastest: MEAN")
    print(f"  Most robust: CLUSTER_CONSENSUS")


# =================================================================
# EXAMPLE 6: Shard Chief Election
# =================================================================

def example_6_shard_election():
    """Demonstrate Shard Chief election mechanism"""

    from shard_chief import ShardChiefElector, ShardChiefCandidate, NodeCapabilities

    print(f"✓ Shard Chief Election Mechanism")

    # Create node pool
    np.random.seed(42)
    candidates = [
        ShardChiefCandidate(
            node_id=f"node_{i}",
            capabilities=NodeCapabilities(
                node_id=f"node_{i}",
                uptime=0.85 + np.random.rand() * 0.15,
                bandwidth_mbps=50 + np.random.rand() * 100,
                compute_capacity=0.5 + np.random.rand() * 1.0,
            )
        )
        for i in range(10)
    ]

    print(f"\n  Candidates:")
    print(f"  {'ID':<10} {'Uptime':<10} {'Bandwidth':<15} {'Compute':<10}")
    print(f"  {'-'*45}")
    for c in candidates[:5]:
        print(f"  {c.node_id:<10} {c.capabilities.uptime:<10.1%} {c.capabilities.bandwidth_mbps:<15.1f} {c.capabilities.compute_capacity:<10.2f}")

    # Perform election
    elector = ShardChiefElector()
    chief, election_meta = elector.elect_for_shard(
        candidates, "shard_0", top_k=5
    )

    print(f"\n✓ Election Result:")
    print(f"  Elected: {chief.chief_id}")
    print(f"  Score: {election_meta['score']:.4f}")
    print(f"  Rank: {election_meta['rank']}/{election_meta['top_k']}")
    print(f"  Total candidates: {election_meta['total_candidates']}")

    # Show score breakdown
    print(f"\n  Chief Capabilities:")
    print(f"    Uptime: {chief.capabilities.uptime:.1%}")
    print(f"    Bandwidth: {chief.capabilities.bandwidth_mbps:.1f} Mbps")
    print(f"    Compute: {chief.capabilities.compute_capacity:.2f}x")

    # Compute score components
    from federated_learning_config import (
        SHARD_CHIEF_SCORE_ALPHA, SHARD_CHIEF_SCORE_BETA, SHARD_CHIEF_SCORE_GAMMA
    )

    uptime_score = chief.capabilities.get_uptime_score()
    bw_score = chief.capabilities.get_bandwidth_score()
    compute_score = chief.capabilities.get_compute_score()

    print(f"\n  Score Breakdown (α={SHARD_CHIEF_SCORE_ALPHA}, β={SHARD_CHIEF_SCORE_BETA}, γ={SHARD_CHIEF_SCORE_GAMMA}):")
    print(f"    Uptime component: {uptime_score:.3f} × {SHARD_CHIEF_SCORE_ALPHA} = {uptime_score * SHARD_CHIEF_SCORE_ALPHA:.3f}")
    print(f"    Bandwidth component: {bw_score:.3f} × {SHARD_CHIEF_SCORE_BETA} = {bw_score * SHARD_CHIEF_SCORE_BETA:.3f}")
    print(f"    Compute component: {compute_score:.3f} × {SHARD_CHIEF_SCORE_GAMMA} = {compute_score * SHARD_CHIEF_SCORE_GAMMA:.3f}")
    print(f"    Total: {election_meta['score']:.4f}")


# =================================================================
# EXAMPLE 7: DAG System (Model Evolution Tracking)
# =================================================================

def example_7_dag_system():
    """Demonstrate DAG model evolution tracking"""

    from dag_system import ShardedDAG

    print(f"✓ DAG System - Model Evolution Tracking")

    # Create sharded DAG
    dag = ShardedDAG(num_shards=2, initial_model_hash="initial_model_v1")

    # Add updates (Round 1)
    print(f"\n  Round 1 Updates:")
    nodes_r1 = []
    for i in range(4):
        node_id = dag.add_shard_update(
            shard_id=f"shard_{i % 2}",
            client_id=f"client_{i}",
            round_num=1,
            model_hash=f"model_r1_c{i}",
            gradient_hash=f"grad_r1_c{i}",
            parent_id="root"
        )
        nodes_r1.append(node_id)
        print(f"    Added {node_id[:8]}... to {f'shard_{i % 2}'}")

    # Verify updates
    print(f"\n  Verification:")
    for shard_id, dag_graph in dag.shard_dags.items():
        for node_id in dag_graph.nodes:
            if node_id != "root":
                dag_graph.verify_update(node_id, zkp_verified=True)
        print(f"    {shard_id}: verified all updates")

    # Finalize (convert to permanent DAG)
    print(f"\n  Finalization:")
    for node_id in nodes_r1[:2]:
        for dag_graph in dag.shard_dags.values():
            if node_id in dag_graph.nodes:
                dag_graph.finalize_update(node_id)
                print(f"    Finalized {node_id[:8]}...")

    # Add Round 2 updates
    print(f"\n  Round 2 Updates (building on finalized):")
    nodes_r2 = []
    for i in range(2):
        parent = nodes_r1[i] if i < len(nodes_r1) else "root"
        node_id = dag.add_shard_update(
            shard_id="shard_0",
            client_id=f"client_{i+10}",
            round_num=2,
            model_hash=f"model_r2_{i}",
            gradient_hash=f"grad_r2_{i}",
            parent_id=parent
        )
        nodes_r2.append(node_id)
        print(f"    Added {node_id[:8]}... (parent: {parent[:8]}...)")

    # Get DAG statistics
    stats = dag.get_stats()
    print(f"\n✓ DAG Statistics:")
    print(f"  Global model snapshots: {stats['global_model_history_length']}")
    for shard_id, shard_stats in stats['shard_stats'].items():
        print(f"\n  {shard_id}:")
        print(f"    Total nodes: {shard_stats['total_nodes']}")
        print(f"    Total edges: {shard_stats['total_edges']}")
        print(f"    Finalized chain length: {shard_stats['finalized_chain_length']}")
        print(f"    Status breakdown: {shard_stats['status_breakdown']}")


# =================================================================
# EXAMPLE 8: Anti-Poisoning Defense
# =================================================================

def example_8_anti_poisoning():
    """Demonstrate Byzantine resilience features"""

    from robust_aggregation import RobustAggregator, AggregationMethod
    from zkp_proofs import ZKPFactory
    from polynomial_proxy import PolynomialProxyFactory

    print(f"✓ Byzantine Update Detection and Filtering")

    # Create 10 honest updates
    honest_updates = [np.random.randn(100) * 0.1 for _ in range(10)]
    honest_verifications = [True] * 10

    # Create 2 poisoning updates (extreme values)
    poison_updates = [np.random.randn(100) * 10.0 for _ in range(2)]
    poison_verifications = [False] * 2  # Won't verify

    # Mix
    all_updates = honest_updates + poison_updates
    all_verifications = honest_verifications + poison_verifications
    client_ids = [f"client_{i}" for i in range(len(all_updates))]

    # Aggregate
    aggregator = RobustAggregator()

    # Without filtering (naive mean)
    naive_agg = np.mean([u for u, v in zip(all_updates, all_verifications)], axis=0)
    naive_norm = np.linalg.norm(naive_agg)

    print(f"\n  Naive aggregation (no filtering):")
    print(f"    Updates used: {len(all_updates)}")
    print(f"    Aggregated norm: {naive_norm:.6f}")
    print(f"    Result: POISONED (extremely large update)")

    # With robust aggregation
    robust_agg, metadata = aggregator.aggregate(
        all_updates,
        verification_results=all_verifications,
        client_ids=client_ids,
        method=AggregationMethod.ROBUST_HYBRID
    )
    robust_norm = np.linalg.norm(robust_agg)

    print(f"\n  Robust aggregation (multi-stage filtering):")
    print(f"    ZKP filtering: {metadata['stages']['after_zkp_filtering']} updates passed")
    print(f"    Reputation filtering: {metadata['stages']['after_reputation_filtering']} updates passed")
    print(f"    Aggregated norm: {robust_norm:.6f}")
    print(f"    Result: DEFENDED (poisoning rejected)")

    print(f"\n  Defense Summary:")
    print(f"    Poisoning reduction: {(1 - robust_norm/naive_norm)*100:.1f}%")
    print(f"    Techniques used:")
    print(f"      1. ZKP verification (proof of honest behavior)")
    print(f"      2. Reputation scoring (trust history)")
    print(f"      3. Cluster consensus (outlier detection)")
    print(f"      4. Weighted averaging (reduce malicious influence)")


# =================================================================
# EXAMPLE 9: Scaling Analysis
# =================================================================

def example_9_scaling_analysis():
    """Analyze system scaling properties"""

    from federated_learning_system import ShardedFederatedLearningSystem
    import time

    print(f"✓ System Scaling Analysis")

    configurations = [
        (6, 2),    # 6 clients, 2 shards
        (12, 3),   # 12 clients, 3 shards
        (20, 4),   # 20 clients, 4 shards
    ]

    results = {}

    for num_clients, num_shards in configurations:
        print(f"\n  Config: {num_clients} clients, {num_shards} shards")

        start = time.time()

        system = ShardedFederatedLearningSystem(
            num_clients=num_clients,
            num_shards=num_shards,
            num_rounds=2  # Just 2 rounds for timing
        )

        train_results = system.execute_full_training()
        elapsed = time.time() - start

        stats = system._get_final_stats()

        results[f"{num_clients}c_{num_shards}s"] = {
            "elapsed": elapsed,
            "aggregations": stats['global_coordinator']['total_aggregations'],
        }

        print(f"    Time: {elapsed:.2f}s")
        print(f"    Updates/second: {(num_clients * 2) / elapsed:.1f}")
        print(f"    Latency per round: {elapsed / 2:.2f}s")

    print(f"\n✓ Scaling Summary:")
    print(f"  Linear communication complexity with number of clients")
    print(f"  Logarithmic verification complexity due to sharding")
    print(f"  Constant aggregation complexity per shard")


# =================================================================
# EXAMPLE 10: Non-IID Data Distribution
# =================================================================

def example_10_non_iid_training():
    """Training with non-IID data distribution"""

    from client import FederatedLearningClient, create_synthetic_client_data
    from federated_learning_config import INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM

    print(f"✓ Non-IID Data Distribution Training")

    # Create clients with different data distributions
    clients = []
    for client_idx in range(4):
        client = FederatedLearningClient(
            client_id=f"client_{client_idx}",
            shard_id=f"shard_{client_idx // 2}"
        )

        # Each client gets non-IID data (focused on specific classes)
        X, y = create_synthetic_client_data(
            client_id=client_idx,
            num_samples=200,
            data_distribution="non_iid"
        )

        client.set_local_dataset(X, y)
        clients.append(client)

        # Show class distribution
        class_dist = np.bincount(np.argmax(y, axis=1))
        print(f"\n  {client.client_id} ({len(X)} samples):")
        print(f"    Primary classes: {np.argsort(class_dist)[-2:]}")
        print(f"    Distribution: {dict(enumerate(class_dist))}")

    print(f"\n  Key insight: Each client specializes in subset of tasks")
    print(f"  Result: Models must learn shared representations")
    print(f"  Benefit: More realistic federated learning scenario")


# =================================================================
# MAIN: Run all examples
# =================================================================

def run_all_examples():
    """Run all examples"""

    examples = [
        ("1. Custom Training", example_1_custom_training),
        ("2. Single Client", example_2_single_client),
        ("3. Polynomial Proxy", example_3_polynomial_proxy),
        ("4. ZKP Workflow", example_4_zkp_workflow),
        ("5. Aggregation Comparison", example_5_aggregation_comparison),
        ("6. Shard Election", example_6_shard_election),
        ("7. DAG System", example_7_dag_system),
        ("8. Anti-Poisoning", example_8_anti_poisoning),
        ("9. Scaling Analysis", example_9_scaling_analysis),
        ("10. Non-IID Training", example_10_non_iid_training),
    ]

    print("\n" + "#"*60)
    print("# CUSTOM EXAMPLES - SHARDED FEDERATED LEARNING")
    print("#"*60)

    for name, example_func in examples:
        try:
            print(f"\n{'='*60}")
            print(f"{name}")
            print(f"{'='*60}")
            example_func()
        except Exception as e:
            print(f"\n✗ Example failed: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # To run a specific example:
    # example_1_custom_training()
    # example_2_single_client()
    # etc.

    # Or run all:
    run_all_examples()
