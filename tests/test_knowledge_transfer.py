import numpy as np

from knowledge_transfer import (
    AGIKnowledgeTransferSystem,
    KnowledgeTransferConfig,
    KnowledgeCategory,
    ProgressiveNetwork,
)


class DummyPhysics:
    def __init__(self):
        self.conservation_enabled = False


class DummyWorldModel:
    def __init__(self, global_dim: int = 16):
        self.global_dim = global_dim
        self.physics = DummyPhysics()


def test_invariance_discovery_energy_conservation():
    wm = DummyWorldModel()
    sys = AGIKnowledgeTransferSystem(wm, config=KnowledgeTransferConfig())

    # Constant energy across examples -> should mark conservation
    data = []
    for _ in range(20):
        slots = [np.ones((4,), dtype=float), -np.ones((4,), dtype=float)]
        data.append({'slots': slots})

    inv = sys.identify_invariances(data)
    assert isinstance(inv, list)
    assert any(i.get('type') == 'conservation' and i.get('quantity') == 'energy' for i in inv)


def test_domain_mapping_fit_and_apply():
    wm = DummyWorldModel()
    cfg = KnowledgeTransferConfig(mapping_ridge=1e-6)
    sys = AGIKnowledgeTransferSystem(wm, config=cfg)

    # y = 2x mapping
    paired = []
    for i in range(10):
        x = np.array([float(i), float(i + 1)], dtype=float)
        y = 2.0 * x
        paired.append({'source_state': x, 'target_state': y})

    ok = sys.learn_domain_mapping('source_to_target', paired)
    assert ok is True

    yhat = sys.apply_domain_mapping('source_to_target', np.array([3.0, 4.0], dtype=float))
    yhat = np.asarray(yhat).reshape(-1)
    assert yhat.shape[0] == 2
    assert np.allclose(yhat, np.array([6.0, 8.0], dtype=float), atol=1e-1)


def test_transfer_to_target_enables_conservation_flag():
    source_wm = DummyWorldModel()
    target_wm = DummyWorldModel()

    sys = AGIKnowledgeTransferSystem(source_wm, config=KnowledgeTransferConfig())

    sys.identify_invariances([{'slots': [np.ones((4,), dtype=float), -np.ones((4,), dtype=float)]} for _ in range(10)])

    out = sys.transfer_to_target(
        target_data=[{'source_state': np.array([1.0, 2.0]), 'target_state': np.array([2.0, 4.0])}],
        target_world_model=target_wm,
    )

    assert isinstance(out, dict)
    assert out.get('num_invariances_transferred', 0) >= 1
    assert target_wm.physics.conservation_enabled is True


def test_knowledge_category_axioms_verification():
    cat = KnowledgeCategory()
    cat.add_domain('A')
    cat.add_domain('B')
    cat.add_morphism('A', 'B', lambda x: x + 1.0)
    cat.add_morphism('B', 'B', lambda x: x * 2.0)
    assert cat.verify_category_axioms(probe=1.0) is True


def test_progressive_network_has_lateral_connections_and_runs():
    pn = ProgressiveNetwork(input_dim=4, hidden_dims=[8, 8], output_dim=4)
    pn.add_column(label='t0')
    pn.add_column(label='t1')

    # second column should have laterals for hidden layers
    assert len(pn.columns) == 2
    assert len(pn.columns[1].lateral_connections) == 2
    assert len(pn.columns[1].lateral_connections[0]) == 1

    from nn import Tensor

    x = Tensor(np.ones((4,), dtype=float))
    y = pn(x)
    assert y.data.shape[0] == 4


def test_kernel_domain_mapping_option():
    wm = DummyWorldModel()
    cfg = KnowledgeTransferConfig(mapping_ridge=1e-3, kernel_mapping=True, kernel_gamma=0.5)
    sys = AGIKnowledgeTransferSystem(wm, config=cfg)

    paired = []
    for i in range(12):
        x = np.array([float(i), float(i + 1)], dtype=float)
        y = np.array([np.sin(float(i)), np.cos(float(i + 1))], dtype=float)
        paired.append({'source_state': x, 'target_state': y})

    ok = sys.learn_domain_mapping('k', paired)
    assert ok is True
    yhat = sys.apply_domain_mapping('k', np.array([3.0, 4.0], dtype=float))
    yhat = np.asarray(yhat).reshape(-1)
    assert yhat.shape[0] == 2
    assert np.isfinite(yhat).all()


def test_concrete_cross_domain_transfer_reduces_error_on_heldout():
    rng = np.random.RandomState(0)

    # Linear cross-domain mapping: y = A x + b + noise
    A = np.array([[1.5, -0.2, 0.0], [0.3, 0.9, 0.1], [-0.1, 0.2, 1.2]], dtype=float)
    b = np.array([0.5, -1.0, 0.25], dtype=float)

    def gen_pair():
        x = rng.randn(3).astype(float)
        y = (A @ x) + b + 0.01 * rng.randn(3)
        return x, y

    train = [gen_pair() for _ in range(64)]
    test = [gen_pair() for _ in range(64)]

    wm = DummyWorldModel(global_dim=3)
    sys = AGIKnowledgeTransferSystem(wm, config=KnowledgeTransferConfig(mapping_ridge=1e-6))

    paired_train = [{'source_state': x, 'target_state': y} for x, y in train]
    assert sys.learn_domain_mapping('lin', paired_train) is True

    # Baseline: identity (no transfer)
    mse_base = 0.0
    mse_map = 0.0
    for x, y in test:
        y0 = x
        y1 = np.asarray(sys.apply_domain_mapping('lin', x)).reshape(-1)
        mse_base += float(np.mean((y0 - y) ** 2))
        mse_map += float(np.mean((y1 - y) ** 2))
    mse_base /= float(len(test))
    mse_map /= float(len(test))

    # Must significantly improve (robust threshold)
    assert mse_map < 0.2 * mse_base

    # Nonlinear cross-domain mapping: y = [sin(x0), cos(x1), x2^2]
    def gen_pair_nl():
        x = rng.uniform(low=-2.0, high=2.0, size=(3,)).astype(float)
        y = np.array([np.sin(x[0]), np.cos(x[1]), x[2] ** 2], dtype=float) + 0.01 * rng.randn(3)
        return x, y

    train_nl = [gen_pair_nl() for _ in range(96)]
    test_nl = [gen_pair_nl() for _ in range(96)]

    sys2 = AGIKnowledgeTransferSystem(
        wm,
        config=KnowledgeTransferConfig(mapping_ridge=1e-2, kernel_mapping=True, kernel_gamma=0.7),
    )
    paired_train_nl = [{'source_state': x, 'target_state': y} for x, y in train_nl]
    assert sys2.learn_domain_mapping('nl', paired_train_nl) is True

    mse_base2 = 0.0
    mse_map2 = 0.0
    for x, y in test_nl:
        y0 = x
        y1 = np.asarray(sys2.apply_domain_mapping('nl', x)).reshape(-1)
        mse_base2 += float(np.mean((y0 - y) ** 2))
        mse_map2 += float(np.mean((y1 - y) ** 2))
    mse_base2 /= float(len(test_nl))
    mse_map2 /= float(len(test_nl))

    assert mse_map2 < 0.5 * mse_base2
