import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from nn import Tensor


@dataclass
class WorldModelEvalConfig:
    seed: int = 0
    num_trials: int = 8
    horizon: int = 5
    num_action_samples: int = 32


def _to_np(x: Any) -> np.ndarray:
    if x is None:
        return np.zeros((1,), dtype=float)
    if hasattr(x, 'data'):
        return np.asarray(x.data, dtype=float)
    return np.asarray(x, dtype=float)


def prediction_mse(pred: Any, target: Any) -> float:
    a = _to_np(pred).reshape(-1)
    b = _to_np(target).reshape(-1)
    n = min(a.size, b.size)
    if n <= 0:
        return 0.0
    return float(np.mean((a[:n] - b[:n]) ** 2))


def uncertainty_nll(error: np.ndarray, logvar: np.ndarray) -> float:
    e = np.asarray(error, dtype=float).reshape(-1)
    lv = np.asarray(logvar, dtype=float).reshape(-1)
    n = min(e.size, lv.size)
    if n <= 0:
        return 0.0
    e = e[:n]
    lv = np.clip(lv[:n], -20.0, 20.0)
    var = np.exp(lv)
    return float(0.5 * np.mean((e * e) / (var + 1e-9) + lv))


class WorldModelEvaluator:
    def __init__(self, facade: Any, config: Optional[WorldModelEvalConfig] = None):
        self.facade = facade
        self.config = config if config is not None else WorldModelEvalConfig()
        self.rng = np.random.RandomState(int(self.config.seed))

    def _random_state(self) -> Tuple[Tensor, Tensor, Tensor]:
        # Use facade.model slot_dim/rel_dim/global_dim if present.
        slot_dim = int(getattr(getattr(self.facade, 'model', None), 'slot_dim', 64))
        rel_dim = int(getattr(getattr(self.facade, 'model', None), 'rel_dim', 32))
        global_dim = int(getattr(getattr(self.facade, 'model', None), 'global_dim', 128))
        num_slots = 6

        slots = self.rng.randn(num_slots, slot_dim).astype(float)
        rels = self.rng.randn(num_slots, num_slots, rel_dim).astype(float)
        g = self.rng.randn(global_dim).astype(float)
        return Tensor(slots), Tensor(rels), Tensor(g)

    def evaluate_prediction(self) -> Dict[str, float]:
        mses: List[float] = []
        nlls: List[float] = []

        for _ in range(int(max(1, self.config.num_trials))):
            slots, rels, g = self._random_state()

            # pseudo-target: small perturbed previous state (since we don't have env ground truth)
            target = Tensor(slots.data + 0.05 * self.rng.randn(*slots.data.shape))

            out = self.facade.predict(slots, rels, g, apply_physics=False, store_memory=False)
            pred_slots = out.get('slots', slots)
            mses.append(prediction_mse(pred_slots, target))

            # If uncertainty is exposed, compute a crude NLL proxy
            logvars = out.get('slot_logvars', None)
            if logvars is not None:
                err = _to_np(pred_slots) - _to_np(target)
                nlls.append(uncertainty_nll(err, _to_np(logvars)))

        return {
            'prediction_mse': float(np.mean(mses)) if mses else 0.0,
            'uncertainty_nll': float(np.mean(nlls)) if nlls else 0.0,
        }

    def evaluate_planning_smoke(self) -> Dict[str, float]:
        # This is a smoke benchmark: ensure plan() works and returns finite cost.
        slots, rels, g = self._random_state()
        goal_slots = Tensor(slots.data + 0.1 * self.rng.randn(*slots.data.shape))

        actions, cost = self.facade.plan(slots, rels, goal_slots, horizon=int(self.config.horizon), num_samples=int(self.config.num_action_samples))
        cost_f = float(cost) if np.isfinite(cost) else float('inf')
        return {
            'planned_steps': float(len(actions) if actions is not None else 0),
            'plan_cost': cost_f,
        }

    def evaluate_transfer_gain(self) -> Dict[str, float]:
        # Concrete transfer gain proxy: mapping/adapter reduces held-out MSE on a known shift.
        A = np.array([[1.2, 0.1], [-0.3, 0.9]], dtype=float)
        b = np.array([0.2, -0.1], dtype=float)

        def gen(n: int):
            xs = self.rng.randn(n, 2).astype(float)
            ys = (xs @ A.T) + b + 0.01 * self.rng.randn(n, 2)
            return xs, ys

        xs_tr, ys_tr = gen(64)
        xs_te, ys_te = gen(64)

        paired = [{'source_state': xs_tr[i], 'target_state': ys_tr[i]} for i in range(xs_tr.shape[0])]

        transfer = getattr(getattr(self.facade, 'model', None), 'transfer', None)
        if transfer is None:
            return {'transfer_gain': 0.0}

        try:
            transfer.train_domain_adapter('eval', paired_data=paired, steps=200, lr=1e-3)
        except Exception:
            pass

        mse_base = float(np.mean((xs_te - ys_te) ** 2))
        try:
            yh = np.stack([np.asarray(transfer.apply_domain_adapter('eval', xs_te[i])).reshape(-1)[:2] for i in range(xs_te.shape[0])], axis=0)
            mse_map = float(np.mean((yh - ys_te) ** 2))
        except Exception:
            mse_map = mse_base

        gain = (mse_base - mse_map) / (mse_base + 1e-9)
        return {'transfer_gain': float(gain)}

    def evaluate_all(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out.update(self.evaluate_prediction())
        out.update(self.evaluate_planning_smoke())
        out.update(self.evaluate_transfer_gain())
        return out
