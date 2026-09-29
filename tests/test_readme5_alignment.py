import numpy as np

from federated_learning_config import (
    LOCAL_EPOCHS,
    LEARNING_RATE,
    COMPRESSION_RATIO,
    QUANTIZE_ENABLED,
    ERROR_FEEDBACK,
)
from federated_learning_system import ShardedFederatedLearningSystem


def test_baseline_alignment_config():
    assert LOCAL_EPOCHS == 5
    assert LEARNING_RATE == 0.05
    assert COMPRESSION_RATIO == 1.0
    assert QUANTIZE_ENABLED is False
    assert ERROR_FEEDBACK is False


def test_round_records_global_eval():
    system = ShardedFederatedLearningSystem(
        num_clients=2,
        num_shards=1,
        num_rounds=1,
        pretrained_model_path=None,
        use_real_data=False,
    )

    # Build a tiny hold-out split from pooled synthetic data and ensure the round logs it.
    round_meta = system.execute_training_round()

    assert "global_eval" in round_meta
    assert 0.0 <= round_meta["global_eval"]["test_accuracy"] <= 1.0
    assert round_meta["global_eval"]["n_test_samples"] > 0
