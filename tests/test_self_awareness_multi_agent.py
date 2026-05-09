import numpy as np

from mind.self_awareness import SelfAwarenessSystem
from mind.multi_agent import MultiAgentMind, MultiAgentConfig


def test_self_awareness_episode_is_non_static_and_updates_over_ticks():
    sa = SelfAwarenessSystem(agent_id='agent_x')

    # two ticks with different introspection should move self_state
    ep1 = sa.observe_tick(
        step=1,
        t_global=0.1,
        brain_state=np.zeros(8, dtype=float),
        action=np.zeros(4, dtype=float),
        reward=0.0,
        done=False,
        introspection={'uncertainty': 0.1, 'prediction_error': 0.1, 'reward': 0.0, 'done': False, 'neuromodulators': {}, 'tool': {'success': 1.0}},
    )
    ep2 = sa.observe_tick(
        step=2,
        t_global=0.2,
        brain_state=np.ones(8, dtype=float) * 0.5,
        action=np.ones(4, dtype=float) * 0.25,
        reward=0.1,
        done=False,
        introspection={'uncertainty': 0.9, 'prediction_error': 0.9, 'reward': 0.1, 'done': False, 'neuromodulators': {'dopamine': 1.0}, 'tool': {'success': 0.0}},
    )

    assert ep1.agent_id == 'agent_x'
    assert ep2.agent_id == 'agent_x'

    # Non-static: self_state should change across different inputs
    assert np.mean(np.abs(ep2.self_state - ep1.self_state)) > 0.0


def test_multi_agent_private_memory_instances_exist():
    ma = MultiAgentMind(config=MultiAgentConfig(num_agents=2, latent_dim=32))
    ids = ma.ids()
    assert 'agent_0' in ids
    assert 'agent_1' in ids

    a0 = ma.get('agent_0')
    a1 = ma.get('agent_1')

    assert a0 is not a1
    assert getattr(a0, 'memory_system', None) is not None
    assert getattr(a1, 'memory_system', None) is not None

    # Private memory objects (distinct instances)
    assert a0.memory_system is not a1.memory_system
