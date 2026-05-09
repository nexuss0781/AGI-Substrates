import numpy as np

from world_model import create_complete_agi_world_model_facade
from world_model_evaluation import WorldModelEvaluator, WorldModelEvalConfig


def test_world_model_evaluation_harness_runs_and_returns_finite_metrics():
    facade = create_complete_agi_world_model_facade(
        slot_dim=32,
        rel_dim=16,
        global_dim=64,
        action_dim=4,
        obs_dim=64,
        memory_system=None,
        vision_encoder=None,
        language_encoder=None,
        with_actions=True,
    )

    evaluator = WorldModelEvaluator(facade, config=WorldModelEvalConfig(seed=0, num_trials=3, horizon=3, num_action_samples=8))
    metrics = evaluator.evaluate_all()

    assert 'prediction_mse' in metrics
    assert 'plan_cost' in metrics
    assert 'transfer_gain' in metrics

    assert np.isfinite(float(metrics['prediction_mse']))
    assert np.isfinite(float(metrics['plan_cost'])) or float(metrics['plan_cost']) == float('inf')
    assert np.isfinite(float(metrics['transfer_gain']))
