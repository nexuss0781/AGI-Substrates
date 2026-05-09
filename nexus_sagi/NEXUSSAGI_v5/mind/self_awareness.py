import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from nn import Tensor, Module, MLP


@dataclass
class SelfAwarenessConfig:
    self_dim: int = 64
    id_dim: int = 32
    ema_state: float = 0.90
    ema_identity: float = 0.98
    lr: float = 1e-3
    agency_lr: float = 1e-3
    grad_clip: float = 10.0
    max_episode_len: int = 256


@dataclass
class SelfEpisode:
    step: int
    t_global: float
    agent_id: str
    self_state: np.ndarray
    identity: np.ndarray
    agency_score: float
    prediction_error: float
    uncertainty: float
    tool_success: float
    tags: List[str]
    payload: Dict[str, Any]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _sgd_step(params: List[Tensor], lr: float, grad_clip: float) -> None:
    lr = float(lr)
    gc = float(grad_clip)
    for p in params:
        g = getattr(p, 'grad', None)
        if g is None:
            continue
        g_arr = np.asarray(getattr(g, 'data', g), dtype=float)
        if g_arr.shape != np.asarray(p.data).shape:
            try:
                g_arr = np.broadcast_to(g_arr, np.asarray(p.data).shape)
            except Exception:
                g_arr = np.zeros_like(p.data)
        if np.isfinite(gc) and gc > 0:
            g_arr = np.clip(g_arr, -gc, gc)
        try:
            p.data = p.data - lr * g_arr
        except Exception:
            continue
        try:
            p.grad = None
        except Exception:
            pass


class SelfAwarenessSystem(Module):
    """Trainable, non-linguistic self-awareness.

    Produces a latent self-state and identity embedding, learns agency attribution,
    and emits compact autobiographical episodes.
    """

    def __init__(self, agent_id: str = 'agent_0', config: Optional[SelfAwarenessConfig] = None):
        self.agent_id = str(agent_id)
        self.config = config if config is not None else SelfAwarenessConfig()

        self._feat_dim = None  # inferred on first tick

        self.self_encoder: Optional[MLP] = None
        self.identity_net: Optional[MLP] = None
        self.agency_action: Optional[MLP] = None
        self.agency_no_action: Optional[MLP] = None

        self.self_state = np.zeros(int(self.config.self_dim), dtype=float)
        self.identity = np.zeros(int(self.config.id_dim), dtype=float)

        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[np.ndarray] = None

        self.episodes: List[SelfEpisode] = []

    def _init_models(self, feat_dim: int, action_dim: int, state_dim: int) -> None:
        self._feat_dim = int(feat_dim)
        self.self_encoder = MLP(int(feat_dim), [128, 128, int(self.config.self_dim)], label=f'self_encoder_{self.agent_id}')
        self.identity_net = MLP(int(self.config.self_dim), [64, int(self.config.id_dim)], label=f'self_id_{self.agent_id}')

        # agency: predict delta(next_state-state)
        self.agency_action = MLP(int(state_dim + action_dim), [128, 128, int(state_dim)], label=f'agency_act_{self.agent_id}')
        self.agency_no_action = MLP(int(state_dim), [128, 128, int(state_dim)], label=f'agency_noact_{self.agent_id}')

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for m in [self.self_encoder, self.identity_net, self.agency_action, self.agency_no_action]:
            if m is not None and hasattr(m, 'parameters'):
                try:
                    params.extend(m.parameters())
                except Exception:
                    pass
        return params

    def _features(self, introspection: Dict[str, Any], state_dim: int, action_dim: int) -> np.ndarray:
        # Stable, numeric, non-linguistic features.
        feats: List[float] = []

        feats.append(_safe_float(introspection.get('uncertainty', 0.0)))
        feats.append(_safe_float(introspection.get('prediction_error', 0.0)))
        feats.append(_safe_float(introspection.get('reward', 0.0)))
        feats.append(1.0 if bool(introspection.get('done', False)) else 0.0)

        nm = introspection.get('neuromodulators', {}) if isinstance(introspection.get('neuromodulators', {}), dict) else {}
        feats.append(_safe_float(nm.get('dopamine', 0.0)))
        feats.append(_safe_float(nm.get('acetylcholine', 0.0)))
        feats.append(_safe_float(nm.get('norepinephrine', 0.0)))
        feats.append(_safe_float(nm.get('plasticity_gate', 0.0)))

        feats.append(_safe_float(introspection.get('attention_confidence', 0.5)))
        feats.append(_safe_float(introspection.get('attention_surprise', 0.0)))

        tool = introspection.get('tool', {}) if isinstance(introspection.get('tool', {}), dict) else {}
        feats.append(_safe_float(tool.get('success', 0.0)))
        feats.append(_safe_float(tool.get('latency_ms', 0.0)) / 1000.0)

        # optional world-model evaluation metrics (if provided)
        wm = introspection.get('world_model_metrics', {}) if isinstance(introspection.get('world_model_metrics', {}), dict) else {}
        feats.append(_safe_float(wm.get('prediction_mse', 0.0)))
        feats.append(_safe_float(wm.get('plan_cost', 0.0)))
        feats.append(_safe_float(wm.get('transfer_gain', 0.0)))

        # pad with low-variance identity of dimensions
        feats.append(float(state_dim) / 256.0)
        feats.append(float(action_dim) / 64.0)

        return np.asarray(feats, dtype=float)

    def get_self_state(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'self_state': self.self_state.copy(),
            'identity': self.identity.copy(),
            'num_episodes': int(len(self.episodes)),
            'last_agency': float(self.episodes[-1].agency_score) if self.episodes else 0.0,
        }

    def _compute_agency(self, prev_state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        if self.agency_action is None or self.agency_no_action is None:
            return 0.0

        s = np.asarray(prev_state, dtype=float).reshape(-1)
        a = np.asarray(action, dtype=float).reshape(-1)
        sp = np.asarray(next_state, dtype=float).reshape(-1)
        delta = sp - s

        # Predict deltas
        pred_a = self.agency_action(Tensor(np.concatenate([s, a], dtype=float))).data
        pred_n = self.agency_no_action(Tensor(s)).data

        err_a = float(np.mean((np.asarray(pred_a).reshape(-1)[: delta.size] - delta) ** 2))
        err_n = float(np.mean((np.asarray(pred_n).reshape(-1)[: delta.size] - delta) ** 2))

        # Agency is how much action-conditioned model improves.
        score = (err_n - err_a) / (err_n + 1e-9)
        return float(np.clip(score, -1.0, 1.0))

    def _train_agency(self, prev_state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        if self.agency_action is None or self.agency_no_action is None:
            return

        s = np.asarray(prev_state, dtype=float).reshape(-1)
        a = np.asarray(action, dtype=float).reshape(-1)
        sp = np.asarray(next_state, dtype=float).reshape(-1)
        delta = sp - s

        # Action-conditioned
        x1 = Tensor(np.concatenate([s, a], dtype=float))
        y = Tensor(delta)
        pred1 = self.agency_action(x1)
        loss1 = ((pred1 - y) * (pred1 - y)).sum()
        for p in self.agency_action.parameters():
            p.grad = np.zeros_like(p.data)
        loss1.backward()
        _sgd_step(self.agency_action.parameters(), lr=float(self.config.agency_lr), grad_clip=float(self.config.grad_clip))

        # No-action
        x0 = Tensor(s)
        pred0 = self.agency_no_action(x0)
        loss0 = ((pred0 - y) * (pred0 - y)).sum()
        for p in self.agency_no_action.parameters():
            p.grad = np.zeros_like(p.data)
        loss0.backward()
        _sgd_step(self.agency_no_action.parameters(), lr=float(self.config.agency_lr), grad_clip=float(self.config.grad_clip))

    def observe_tick(
        self,
        step: int,
        t_global: float,
        brain_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        introspection: Dict[str, Any],
    ) -> SelfEpisode:
        s = np.asarray(brain_state, dtype=float).reshape(-1)
        a = np.asarray(action, dtype=float).reshape(-1)

        feat = self._features(introspection=introspection, state_dim=int(s.size), action_dim=int(a.size))
        if self.self_encoder is None:
            self._init_models(feat_dim=int(feat.size), action_dim=int(a.size), state_dim=int(s.size))

        # Train self encoder to predict a stable embedding that tracks introspection.
        # Objective: minimize change while remaining sensitive to features.
        x = Tensor(feat)
        z = self.self_encoder(x) if self.self_encoder is not None else Tensor(np.zeros(int(self.config.self_dim)))
        z_np = np.asarray(z.data, dtype=float).reshape(-1)

        # Non-static: update via EMA + feature-conditioned output.
        self.self_state = float(self.config.ema_state) * self.self_state + (1.0 - float(self.config.ema_state)) * z_np[: self.self_state.size]

        # Identity: slow EMA of identity_net(self_state)
        if self.identity_net is not None:
            id_pred = self.identity_net(Tensor(self.self_state)).data
            id_np = np.asarray(id_pred, dtype=float).reshape(-1)
            self.identity = float(self.config.ema_identity) * self.identity + (1.0 - float(self.config.ema_identity)) * id_np[: self.identity.size]

        # Agency score (uses previous transition)
        agency = 0.0
        if self._prev_state is not None and self._prev_action is not None:
            try:
                self._train_agency(self._prev_state, self._prev_action, s)
                agency = self._compute_agency(self._prev_state, self._prev_action, s)
            except Exception:
                agency = 0.0

        self._prev_state = s.copy()
        self._prev_action = a.copy()

        pred_err = _safe_float(introspection.get('prediction_error', 0.0))
        unc = _safe_float(introspection.get('uncertainty', 0.0))
        tool_success = _safe_float(introspection.get('tool', {}).get('success', 0.0)) if isinstance(introspection.get('tool', {}), dict) else 0.0

        tags = ['self_awareness', f'agent:{self.agent_id}']
        if done:
            tags.append('episode_end')

        payload = {
            'reward': float(reward),
            'done': bool(done),
            'introspection': introspection,
        }

        ep = SelfEpisode(
            step=int(step),
            t_global=float(t_global),
            agent_id=self.agent_id,
            self_state=self.self_state.copy(),
            identity=self.identity.copy(),
            agency_score=float(agency),
            prediction_error=float(pred_err),
            uncertainty=float(unc),
            tool_success=float(tool_success),
            tags=tags,
            payload=payload,
        )

        self.episodes.append(ep)
        if len(self.episodes) > int(self.config.max_episode_len):
            self.episodes = self.episodes[-int(self.config.max_episode_len) :]

        return ep

    def to_memory_item(self, episode: SelfEpisode) -> Dict[str, Any]:
        # Compact memory payload; non-static because it contains evolving self_state/identity and errors.
        return {
            'agent_id': episode.agent_id,
            'step': int(episode.step),
            't_global': float(episode.t_global),
            'self_state': episode.self_state.astype(float),
            'identity': episode.identity.astype(float),
            'agency': float(episode.agency_score),
            'prediction_error': float(episode.prediction_error),
            'uncertainty': float(episode.uncertainty),
            'tool_success': float(episode.tool_success),
            'tags': list(episode.tags),
        }
