from agi_training import AGITrainer, AGITrainingConfig


def test_agi_trainer_runs_minimal_loop():
    trainer = AGITrainer(config=AGITrainingConfig(seed=0, rollout_steps=3, adapter_train_steps=5, adapter_lr=1e-3))
    metrics = trainer.train(num_iterations=2, checkpoint_path=None)

    assert metrics['iterations'] == 2
    assert metrics['buffer_size'] >= 1

    # Probe must return a boolean-like ok
    assert all('probe_ok' in p for p in metrics['probe'])
