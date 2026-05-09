import sys
import os
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import bio_neural_substrate as bns


def test_lif_spikes():
    pop = bns.LIFPopulation(n=10, tau_m=10.0, v_rest=-70.0, v_reset=-70.0, v_th=-55.0, r_m=1.0, refractory_ms=2.0)
    pop.reset_state()
    dt = 1.0

    spikes_any = False
    for _ in range(100):
        s = pop.step(np.ones(10) * 50.0, dt_ms=dt)
        if np.sum(s) > 0:
            spikes_any = True
            break

    assert spikes_any, "LIFPopulation produced no spikes under strong constant current"


def test_hodgkin_huxley_spikes():
    pop = bns.HodgkinHuxleyPopulation(n=5)
    pop.reset_state()
    dt = 0.01

    spikes_total = 0.0
    for _ in range(5000):
        spikes, _v = pop.step(np.ones(5) * 10.0, dt_ms=dt)
        spikes_total += float(np.sum(spikes))

    assert spikes_total > 0.0, "HodgkinHuxleyPopulation produced no spikes under tonic input"


def test_izhikevich_spikes():
    pop = bns.IzhikevichPopulation(n=5)
    pop.reset_state()
    spikes_total = 0.0
    for _ in range(200):
        spikes_total += float(np.sum(pop.step(np.ones(5) * 10.0, dt_ms=1.0)))
    assert spikes_total > 0.0, "IzhikevichPopulation produced no spikes under tonic input"


def test_neural_substrate_context_fusion_runs():
    ns = bns.NeuralSubstrate(evidence_dim=8, latent_dim=16, dt_ms=1.0, seed=0)
    latent0, stats0 = ns.forward(evidence=np.ones(8), modulators=None, context=None)
    latent1, stats1 = ns.forward(evidence=np.ones(8), modulators=None, context=np.ones(8))
    assert latent0.shape == latent1.shape
    assert 'context_gain' in stats1


def test_small_world_metrics():
    adj = bns.watts_strogatz_adjacency(n=100, k=10, p=0.05, rng=np.random.RandomState(0))
    sigma = bns.small_world_index(adj, rng=np.random.RandomState(1))
    assert sigma >= 0.0


def test_avalanches_and_powerlaw_fit_runs():
    out = bns.run_column_simulation(steps=200, dt_ms=1.0, input_scale=10.0, seed=0)
    assert 'avalanches' in out
    assert 'avalanche_alpha' in out
    assert out['avalanche_alpha'] >= 0.0


def test_pair_stdp_directionality():
    n_pre, n_post = 1, 1
    stdp = bns.PairSTDP(n_pre=n_pre, n_post=n_post)
    w = np.zeros((n_post, n_pre), dtype=np.float64)

    # pre at t=0, post at t=10 => LTP
    w = stdp.update(w, s_pre=np.array([1.0]), s_post=np.array([0.0]), t_ms=0.0)
    w = stdp.update(w, s_pre=np.array([0.0]), s_post=np.array([1.0]), t_ms=10.0)
    w_ltp = float(w[0, 0])

    # reset and do LTD: post at t=0, pre at t=10
    stdp = bns.PairSTDP(n_pre=n_pre, n_post=n_post)
    w2 = np.zeros((n_post, n_pre), dtype=np.float64)
    w2 = stdp.update(w2, s_pre=np.array([0.0]), s_post=np.array([1.0]), t_ms=0.0)
    w2 = stdp.update(w2, s_pre=np.array([1.0]), s_post=np.array([0.0]), t_ms=10.0)
    w_ltd = float(w2[0, 0])

    assert w_ltp >= w_ltd, "PairSTDP did not show expected causal asymmetry (pre-before-post > post-before-pre)"


if __name__ == "__main__":
    test_lif_spikes()
    test_hodgkin_huxley_spikes()
    test_izhikevich_spikes()
    test_small_world_metrics()
    test_avalanches_and_powerlaw_fit_runs()
    test_pair_stdp_directionality()
    test_neural_substrate_context_fusion_runs()
    print("[PASSED] bio_neural_substrate basic tests")
