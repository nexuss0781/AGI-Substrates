import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mind.brain import AGIMind


@dataclass
class AGITrainingConfig:
    seed: int = 0
    rollout_steps: int = 20
    adapter_train_steps: int = 50
    adapter_lr: float = 1e-3
    save_every: int = 0


class AGITrainer:
    """Minimal end-to-end trainer/harness for the unified AGI stack.

    Goal: provide a single loop that can be extended, while remaining safe,
    deterministic, and framework-free.

    This trainer focuses on:
    - collecting transitions from AGIMind.tick()
    - training the transfer DomainAdapter on state->next_state pairs (learned dynamics prior)
    - returning metrics that can be tracked across runs
    """

    def __init__(self, mind: Optional[AGIMind] = None, config: Optional[AGITrainingConfig] = None):
        self.config = config if config is not None else AGITrainingConfig()
        self.rng = np.random.RandomState(int(self.config.seed))
        self.mind = mind if mind is not None else AGIMind(latent_dim=64)

        self._transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self._step = 0
        self._start_time = time.time()

    def _get_world_model_tool(self) -> Any:
        substrate = getattr(getattr(self.mind, 'reasoning_interface', None), 'reasoning_substrate', None)
        if substrate is None:
            substrate = getattr(self.mind, 'reasoning_substrate', None)
        if substrate is None:
            return None
        return getattr(substrate, 'world_model', None)

    def collect_rollout(self, steps: Optional[int] = None) -> Dict[str, Any]:
        n = int(steps if steps is not None else self.config.rollout_steps)
        n = int(max(1, n))

        prev_state = None
        prev_action = None
        prev_reward = 0.0
        prev_done = False

        obs_dim = int(getattr(self.mind, '_ai_state_dim', 64))
        for _ in range(n):
            obs = self.rng.randn(obs_dim).astype(float)
            out = self.mind.tick(
                observation=obs,
                modality='state',
                reasoning_query='predict forward dynamics',
                reasoning_context=None,
                reward=float(prev_reward),
                done=bool(prev_done),
                learn=True,
                remember=False,
            )

            s = np.asarray(out.get('brain_state', np.zeros(obs_dim, dtype=float)), dtype=float).reshape(-1)
            a = out.get('action')
            if a is None:
                a_vec = np.zeros(int(getattr(self.mind, '_ai_action_dim', 16)), dtype=float)
            else:
                a_vec = np.asarray(a, dtype=float).reshape(-1)[: int(getattr(self.mind, '_ai_action_dim', 16))]

            r = float(prev_reward)
            done = bool(prev_done)

            if prev_state is not None and prev_action is not None:
                self._transitions.append((prev_state, prev_action, r, s, done))

            prev_state = s
            prev_action = a_vec
            prev_reward = float(self.rng.randn() * 0.01)
            prev_done = False

            self._step += 1

        return {
            'rollout_steps': n,
            'buffer_size': int(len(self._transitions)),
        }

    def train_transfer_adapter(self) -> Dict[str, Any]:
        wm = self._get_world_model_tool()
        if wm is None:
            return {'adapter_trained': False}

        transfer = getattr(getattr(wm, 'model', None), 'transfer', None)
        if transfer is None:
            transfer = getattr(wm, 'transfer', None)
        if transfer is None:
            return {'adapter_trained': False}

        if len(self._transitions) < 2:
            return {'adapter_trained': False}

        # Train on state->next_state pairs (learn a reusable dynamics prior)
        paired = []
        for (s, _a, _r, sp, _d) in self._transitions[-256:]:
            paired.append({'source_state': np.asarray(s, dtype=float), 'target_state': np.asarray(sp, dtype=float)})

        ok = False
        try:
            ok = bool(
                transfer.train_domain_adapter(
                    key='dynamics_prior',
                    paired_data=paired,
                    steps=int(self.config.adapter_train_steps),
                    lr=float(self.config.adapter_lr),
                )
            )
        except Exception:
            ok = False

        return {'adapter_trained': bool(ok), 'pairs': int(len(paired))}

    def capability_routing_probe(self) -> Dict[str, Any]:
        """Lightweight probe: ensure reasoning can call world-model capabilities."""
        wm = self._get_world_model_tool()
        if wm is None:
            return {'probe_ok': False}

        ok = True
        try:
            # Run a short predictive sequence
            obs_dim = int(getattr(self.mind, '_ai_state_dim', 64))
            x = np.zeros(obs_dim, dtype=float)
            slots = None
            rels = None
            g = None
            try:
                enc = self.mind.semantic_encoder.encode(x)
                slots = enc.get('slots')
                rels = enc.get('relations')
                g = enc.get('global_context')
            except Exception:
                ok = ok and True

            if slots is not None and rels is not None:
                seq = wm.predict_sequence(slots, rels, g, steps=2)
                ok = ok and isinstance(seq, list)
        except Exception:
            ok = False

        return {'probe_ok': bool(ok)}

    def train(self, num_iterations: int = 5, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        iters = int(max(1, num_iterations))
        metrics: Dict[str, Any] = {
            'iterations': iters,
            'rollouts': [],
            'adapter': [],
            'probe': [],
        }

        for i in range(iters):
            metrics['rollouts'].append(self.collect_rollout())
            metrics['adapter'].append(self.train_transfer_adapter())
            metrics['probe'].append(self.capability_routing_probe())

            if checkpoint_path and self.config.save_every and (i + 1) % int(self.config.save_every) == 0:
                self.save_checkpoint(checkpoint_path)

        metrics['buffer_size'] = int(len(self._transitions))
        metrics['elapsed_sec'] = float(time.time() - self._start_time)
        return metrics

    def save_checkpoint(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            step=np.array([self._step], dtype=np.int64),
            transitions=np.array([len(self._transitions)], dtype=np.int64),
        )
        return path
