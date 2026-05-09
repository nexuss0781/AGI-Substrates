import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nn import Tensor, Module, MLP, Linear, AdaptiveNorm, tensor_concat

try:
    from active_inference_upgrades import MAMLMetaLearner as _MAMLMetaLearner
    _MAML_AVAILABLE = True
except Exception:
    _MAML_AVAILABLE = False
    _MAMLMetaLearner = None

try:
    from memory import AGIMemorySystem as _AGIMemorySystem
    _MEM_AVAILABLE = True
except Exception:
    _MEM_AVAILABLE = False
    _AGIMemorySystem = None


class KnowledgeCategory:
    def __init__(self):
        self.objects: List[str] = []
        self.morphisms: Dict[Tuple[str, str], List[Callable[[Any], Any]]] = {}
        self.identities: Dict[str, Callable[[Any], Any]] = {}

    def add_domain(self, name: str) -> None:
        if name not in self.objects:
            self.objects.append(name)
            self.identities[name] = lambda x: x
            self.morphisms.setdefault((name, name), []).append(self.identities[name])

    def add_morphism(self, source: str, target: str, transform: Callable[[Any], Any]) -> None:
        self.add_domain(source)
        self.add_domain(target)
        self.morphisms.setdefault((source, target), []).append(transform)

    def compose(self, f: Callable[[Any], Any], g: Callable[[Any], Any]) -> Callable[[Any], Any]:
        return lambda x: g(f(x))

    def tensor_product(self, f: Callable[[Any], Any], g: Callable[[Any], Any]) -> Callable[[Tuple[Any, Any]], Tuple[Any, Any]]:
        """Parallel composition: (f ⊗ g)(x, y) = (f(x), g(y))."""
        return lambda xy: (f(xy[0]), g(xy[1]))

    def functor(self, mapping: Dict[str, str], morphism_map: Optional[Callable[[Callable], Callable]] = None) -> Dict[str, Any]:
        """Minimal functor hook: map object names and optionally map morphisms."""
        out = {'objects': {}, 'morphisms': {}}
        for obj in self.objects:
            out['objects'][obj] = mapping.get(obj, obj)
        if morphism_map is None:
            return out
        for (s, t), morphs in self.morphisms.items():
            out['morphisms'][(mapping.get(s, s), mapping.get(t, t))] = [morphism_map(m) for m in morphs]
        return out

    def natural_transformation(self, F: Dict[str, Any], G: Dict[str, Any], eta: Callable[[str, Any], Any]) -> Callable[[str, Any], Any]:
        """Natural transformation hook: eta(obj, x) where x lives in F(obj)."""
        _ = (F, G)
        return eta

    def verify_category_axioms(self, probe: Any = 1.234, atol: float = 1e-9) -> bool:
        """Best-effort verification: identity and associativity on a probe value."""
        # Identity
        for (s, t), morphs in self.morphisms.items():
            if not morphs:
                continue
            id_s = self.identities.get(s, None)
            id_t = self.identities.get(t, None)
            if id_s is None or id_t is None:
                return False
            f = morphs[0]
            try:
                left = self.compose(id_s, f)(probe)
                right = self.compose(f, id_t)(probe)
                base = f(probe)
                if isinstance(base, np.ndarray):
                    if not np.allclose(left, base, atol=atol) or not np.allclose(right, base, atol=atol):
                        return False
                else:
                    if abs(float(left) - float(base)) > atol or abs(float(right) - float(base)) > atol:
                        return False
            except Exception:
                # If callable isn't probe-compatible, skip.
                continue

        # Associativity on any available triple
        for (a, b), fs in self.morphisms.items():
            for (b2, c), gs in self.morphisms.items():
                if b2 != b:
                    continue
                for (c2, d), hs in self.morphisms.items():
                    if c2 != c:
                        continue
                    try:
                        f, g, h = fs[0], gs[0], hs[0]
                        lhs = self.compose(f, self.compose(g, h))(probe)
                        rhs = self.compose(self.compose(f, g), h)(probe)
                        if isinstance(lhs, np.ndarray):
                            if not np.allclose(lhs, rhs, atol=atol):
                                return False
                        else:
                            if abs(float(lhs) - float(rhs)) > atol:
                                return False
                        return True
                    except Exception:
                        continue
        return True

    def get_morphisms(self, source: str, target: str) -> List[Callable[[Any], Any]]:
        return self.morphisms.get((source, target), [])


@dataclass
class Invariance:
    type: str
    quantity: str
    tolerance: float


class InvarianceDiscovery:
    def identify(self, source_data: List[Dict[str, Any]]) -> List[Invariance]:
        energies: List[float] = []
        for ex in source_data:
            slots = ex.get('slots', None)
            if slots is None:
                continue
            e = 0.0
            for s in slots:
                s = np.asarray(s, dtype=float)
                e += float(np.sum(s * s))
            energies.append(e)

        invariances: List[Invariance] = []
        if energies:
            m = float(np.mean(energies))
            sd = float(np.std(energies))
            if abs(m) > 1e-9 and sd < 0.1 * abs(m):
                invariances.append(Invariance(type='conservation', quantity='energy', tolerance=sd))

        # Momentum-like invariance if velocity estimates are present.
        momenta: List[float] = []
        for ex in source_data:
            v = ex.get('slot_velocities', None)
            if v is None:
                continue
            try:
                v = np.asarray(v, dtype=float)
                momenta.append(float(np.sum(v)))
            except Exception:
                continue
        if momenta:
            m = float(np.mean(momenta))
            sd = float(np.std(momenta))
            if abs(m) > 1e-9 and sd < 0.1 * abs(m):
                invariances.append(Invariance(type='conservation', quantity='momentum_proxy', tolerance=sd))

        return invariances

    def identify_lie_symmetries(self, data: List[Dict[str, Any]]) -> List[Invariance]:
        """Best-effort symmetry discovery.

        Detects lightweight invariances that are commonly transferable:
        - approximate sign symmetry: x and -x appear equally likely (mean near 0)
        - approximate shift symmetry: features are stable up to a global offset
        """
        xs: List[np.ndarray] = []
        for ex in data:
            s = ex.get('source_state', None)
            if s is None:
                continue
            try:
                xs.append(np.asarray(s, dtype=float).reshape(-1))
            except Exception:
                continue
        if not xs:
            return []

        X = np.stack(xs, axis=0)
        invs: List[Invariance] = []

        # Sign symmetry proxy: global mean close to 0 relative to std.
        mu = np.mean(X, axis=0)
        sd = np.std(X, axis=0) + 1e-9
        ratio = float(np.mean(np.abs(mu) / sd))
        if ratio < 0.1:
            invs.append(Invariance(type='symmetry', quantity='sign_flip_proxy', tolerance=ratio))

        # Shift symmetry proxy: sample-to-sample mean shifts are small.
        sample_means = np.mean(X, axis=1)
        shift_sd = float(np.std(sample_means))
        overall_sd = float(np.std(X)) + 1e-9
        if shift_sd < 0.1 * overall_sd:
            invs.append(Invariance(type='symmetry', quantity='global_shift_proxy', tolerance=shift_sd))

        return invs

    def identify_differential_invariants(self, trajectories: List[np.ndarray]) -> List[Invariance]:
        """Detect simple differential invariants from trajectories.

        Uses finite differences and checks for approximately conserved norms.
        """
        invs: List[Invariance] = []
        if not trajectories:
            return invs

        speeds: List[float] = []
        accs: List[float] = []
        for tr in trajectories:
            try:
                x = np.asarray(tr, dtype=float)
                if x.ndim != 2 or x.shape[0] < 4:
                    continue
                v = np.diff(x, axis=0)
                a = np.diff(v, axis=0)
                speeds.extend([float(np.linalg.norm(vt)) for vt in v])
                accs.extend([float(np.linalg.norm(at)) for at in a])
            except Exception:
                continue

        if speeds:
            m = float(np.mean(speeds))
            sd = float(np.std(speeds))
            if abs(m) > 1e-9 and sd < 0.1 * abs(m):
                invs.append(Invariance(type='differential', quantity='speed_norm', tolerance=sd))

        if accs:
            m = float(np.mean(accs))
            sd = float(np.std(accs))
            if abs(m) > 1e-9 and sd < 0.1 * abs(m):
                invs.append(Invariance(type='differential', quantity='acc_norm', tolerance=sd))

        return invs


class DomainMapping:
    def __init__(self, ridge: float = 1e-4):
        self.ridge = float(ridge)
        self.mapping: Optional[np.ndarray] = None
        self.source_mean: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> bool:
        if source_features.size == 0 or target_features.size == 0:
            return False
        if source_features.ndim != 2 or target_features.ndim != 2:
            return False
        if source_features.shape[0] != target_features.shape[0]:
            return False

        xs = np.asarray(source_features, dtype=float)
        yt = np.asarray(target_features, dtype=float)

        self.source_mean = xs.mean(axis=0, keepdims=True)
        self.target_mean = yt.mean(axis=0, keepdims=True)
        xs0 = xs - self.source_mean
        yt0 = yt - self.target_mean

        xtx = xs0.T @ xs0
        xtx = xtx + self.ridge * np.eye(xtx.shape[0], dtype=float)
        self.mapping = np.linalg.pinv(xtx) @ xs0.T @ yt0
        return True

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.mapping is None:
            return np.asarray(x, dtype=float)
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        xm = x
        if self.source_mean is not None:
            xm = x - self.source_mean
        y = xm @ self.mapping
        if self.target_mean is not None:
            y = y + self.target_mean
        return y


class DomainAdapter(Module):
    """Trainable domain adapter: learns a mapping into a shared transfer space.

    This is the AGI-grade bridge that enables transfer without relying purely on
    closed-form linear regression. It is intentionally lightweight to preserve
    interpretability and integration with the custom autograd substrate.
    """

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, label: str = 'domain_adapter'):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        hs = [int(h) for h in hidden_dims]
        self.net = MLP(self.in_dim, hs + [self.out_dim], label=label)

    def __call__(self, x: Tensor) -> Tensor:
        return self.net(x)

    def parameters(self) -> List[Tensor]:
        return self.net.parameters()


def _sgd_step(params: Sequence[Tensor], lr: float = 1e-3, grad_clip: float = 10.0) -> None:
    lr = float(lr)
    for p in params:
        g = getattr(p, 'grad', None)
        if g is None:
            continue
        g_arr = np.asarray(g, dtype=float)
        if grad_clip is not None:
            g_arr = np.clip(g_arr, -float(grad_clip), float(grad_clip))
        try:
            p.data = p.data - lr * g_arr
        except Exception:
            continue
        try:
            p.grad = None
        except Exception:
            pass


class RiemannianKnowledgeTransfer:
    """Fisher-diagonal Riemannian approximation (natural-gradient manifold).

    We avoid SciPy/geodesic ODE solving while still providing a true
    metric-aware path: Exp_x(v) ≈ x + F^{-1} v, where F is Fisher diag.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def exponential_map(self, x: np.ndarray, v: np.ndarray, fisher_diag: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        if fisher_diag is None:
            return x + v
        f = np.asarray(fisher_diag, dtype=float)
        if f.shape != x.shape:
            f = np.ones_like(x)
        step = v / (f + self.eps)
        return x + step

    def logarithmic_map(self, x: np.ndarray, y: np.ndarray, fisher_diag: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        delta = y - x
        if fisher_diag is None:
            return delta
        f = np.asarray(fisher_diag, dtype=float)
        if f.shape != x.shape:
            f = np.ones_like(x)
        return delta * (f + self.eps)

    def geodesic_interpolation(self, source_params: np.ndarray, target_params: np.ndarray, t: float,
                               fisher_diag: Optional[np.ndarray] = None) -> np.ndarray:
        v = self.logarithmic_map(np.asarray(source_params, dtype=float), np.asarray(target_params, dtype=float), fisher_diag=fisher_diag)
        return self.exponential_map(np.asarray(source_params, dtype=float), float(t) * v, fisher_diag=fisher_diag)


class NonLinearDomainMapping:
    """Kernel ridge mapping with optional distribution matching penalty (MMD proxy)."""

    def __init__(self, kernel: str = 'rbf', ridge: float = 1e-4, gamma: Optional[float] = None):
        self.kernel = str(kernel)
        self.ridge = float(ridge)
        self.gamma = gamma
        self.Xs: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None

    def _rbf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if self.gamma is None:
            self.gamma = 1.0 / max(1.0, float(X.shape[1]))
        x2 = np.sum(X * X, axis=1, keepdims=True)
        y2 = np.sum(Y * Y, axis=1, keepdims=True).T
        d2 = x2 - 2.0 * (X @ Y.T) + y2
        return np.exp(-float(self.gamma) * np.clip(d2, 0.0, 1e12))

    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.kernel == 'rbf':
            return self._rbf(X, Y)
        # linear fallback
        return np.asarray(X, dtype=float) @ np.asarray(Y, dtype=float).T

    def fit(self, X_s: np.ndarray, Y_t: np.ndarray) -> bool:
        X_s = np.asarray(X_s, dtype=float)
        Y_t = np.asarray(Y_t, dtype=float)
        if X_s.ndim != 2 or Y_t.ndim != 2 or X_s.shape[0] != Y_t.shape[0]:
            return False

        self.Xs = X_s
        self.target_mean = Y_t.mean(axis=0, keepdims=True)
        Y0 = Y_t - self.target_mean

        K = self._kernel(X_s, X_s)
        n = int(K.shape[0])
        A = K + self.ridge * np.eye(n, dtype=float)
        try:
            self.alpha = np.linalg.solve(A, Y0)
        except Exception:
            self.alpha = np.linalg.pinv(A) @ Y0
        return True

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.Xs is None or self.alpha is None:
            return np.asarray(x, dtype=float)
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        Kx = self._kernel(x, self.Xs)
        y = Kx @ self.alpha
        if self.target_mean is not None:
            y = y + self.target_mean
        return y


class ProgressiveColumn(Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int,
                 num_prev_columns: int = 0, label: str = 'prog'):
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.output_dim = int(output_dim)

        self.num_prev_columns = int(num_prev_columns)

        dims = [self.input_dim] + list(self.hidden_dims) + [self.output_dim]
        layers: List[Module] = []
        # Lateral connections for each hidden stage (exclude final projection).
        # Shape: [num_hidden_layers][num_prev_columns]
        self.lateral_connections: List[List[Linear]] = []
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1], label=f'{label}_l{i}'))
            if i < len(dims) - 2:
                layers.append(AdaptiveNorm(dims[i + 1], label=f'{label}_n{i}'))

                laterals: List[Linear] = []
                for j in range(self.num_prev_columns):
                    laterals.append(Linear(dims[i + 1], dims[i + 1], label=f'{label}_lat_{j}_l{i}'))
                self.lateral_connections.append(laterals)
        self.layers = layers

    def forward(self, x: Tensor, prev_column_hiddens: Optional[List[List[Tensor]]] = None) -> Tuple[Tensor, List[Tensor]]:
        h = x
        hidden_states: List[Tensor] = []

        # We index lateral connections by hidden-layer index (linear->norm pairs)
        hidden_layer_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                h = layer(h)
                # apply laterals only for hidden layers (not final projection)
                if hidden_layer_idx < len(self.lateral_connections) and prev_column_hiddens:
                    lateral_sum = Tensor(np.zeros_like(h.data))
                    for j, lat in enumerate(self.lateral_connections[hidden_layer_idx]):
                        if j < len(prev_column_hiddens) and hidden_layer_idx < len(prev_column_hiddens[j]):
                            lateral_sum = lateral_sum + lat(prev_column_hiddens[j][hidden_layer_idx])
                    h = h + lateral_sum
            elif isinstance(layer, AdaptiveNorm):
                h = layer(h).relu()
                hidden_states.append(h)
                hidden_layer_idx += 1
            else:
                h = layer(h)
        return h, hidden_states

    def __call__(self, x: Tensor, prev_column_hiddens: Optional[List[List[Tensor]]] = None) -> Tensor:
        out, _ = self.forward(x, prev_column_hiddens=prev_column_hiddens)
        return out

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


class ProgressiveNetwork(Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.output_dim = int(output_dim)
        self.columns: List[ProgressiveColumn] = []

    def add_column(self, label: str = 'task') -> ProgressiveColumn:
        col = ProgressiveColumn(
            self.input_dim,
            self.hidden_dims,
            self.output_dim,
            num_prev_columns=len(self.columns),
            label=f'prog_{label}_{len(self.columns)}',
        )
        self.columns.append(col)
        return col

    def __call__(self, x: Tensor) -> Tensor:
        if not self.columns:
            self.add_column(label='default')

        # Compute previous columns' hidden states for lateral connections.
        prev_hiddens: List[List[Tensor]] = []
        for i in range(len(self.columns) - 1):
            _, hs = self.columns[i].forward(x, prev_column_hiddens=prev_hiddens if prev_hiddens else None)
            prev_hiddens.append(hs)
        return self.columns[-1](x, prev_column_hiddens=prev_hiddens if prev_hiddens else None)

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for c in self.columns:
            params.extend(c.parameters())
        return params


class EWCConsolidator:
    def __init__(self, model: Any, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = float(ewc_lambda)
        self.fisher_information: Dict[int, np.ndarray] = {}
        self.optimal_params: Dict[int, np.ndarray] = {}

    def snapshot_params(self, params: Sequence[Tensor]) -> None:
        for p in params:
            self.optimal_params[id(p)] = p.data.copy()

    def compute_fisher_diag(self, params: Sequence[Tensor], losses: Sequence[Tensor]) -> None:
        """Monte Carlo style expectation over provided loss samples."""
        fisher: Dict[int, np.ndarray] = {id(p): np.zeros_like(p.data) for p in params}
        n = max(1, int(len(losses)))

        for loss in losses:
            for p in params:
                p.grad = np.zeros_like(p.data)
            loss.backward()
            for p in params:
                if p.grad is not None:
                    fisher[id(p)] += (p.grad ** 2)

        for pid in fisher:
            fisher[pid] = fisher[pid] / float(n)
        self.fisher_information = fisher

    def penalty(self, params: Sequence[Tensor]) -> float:
        total = 0.0
        for p in params:
            pid = id(p)
            f = self.fisher_information.get(pid, None)
            opt = self.optimal_params.get(pid, None)
            if f is None or opt is None:
                continue
            diff = p.data - opt
            total += float(np.sum(f * (diff ** 2)))
        return 0.5 * self.ewc_lambda * total


@dataclass
class KnowledgeTransferConfig:
    mapping_ridge: float = 1e-4
    ewc_lambda: float = 1000.0
    progressive_hidden: Tuple[int, ...] = (128, 128)
    meta_inner_lr: float = 0.01
    meta_outer_lr: float = 0.001
    kernel_mapping: bool = False
    kernel_gamma: Optional[float] = None
    seed: int = 0


class AGIKnowledgeTransferSystem:
    def __init__(self, source_world_model: Any, config: Optional[KnowledgeTransferConfig] = None):
        self.source_model = source_world_model
        self.config = config if config is not None else KnowledgeTransferConfig()

        self._rng = np.random.RandomState(int(getattr(self.config, 'seed', 0)))

        self.category = KnowledgeCategory()
        self.invariance_discovery = InvarianceDiscovery()
        self.riemannian = RiemannianKnowledgeTransfer()

        self.invariances: List[Invariance] = []
        self.domain_mappings: Dict[str, DomainMapping] = {}

        # Trainable domain adapters (learned transfer, not just regression)
        self.domain_adapters: Dict[str, DomainAdapter] = {}

        self.progressive = ProgressiveNetwork(
            input_dim=int(getattr(source_world_model, 'global_dim', 128)),
            hidden_dims=list(self.config.progressive_hidden),
            output_dim=int(getattr(source_world_model, 'global_dim', 128)),
        )

        self.ewc = EWCConsolidator(model=source_world_model, ewc_lambda=self.config.ewc_lambda)

        self.meta_learner = None
        if _MAML_AVAILABLE:
            try:
                self.meta_learner = _MAMLMetaLearner(
                    model_dim=int(getattr(source_world_model, 'global_dim', 128)),
                    inner_lr=float(self.config.meta_inner_lr),
                    outer_lr=float(self.config.meta_outer_lr),
                )
            except Exception:
                self.meta_learner = None

        self.memory = None
        if _MEM_AVAILABLE:
            try:
                self.memory = _AGIMemorySystem()
            except Exception:
                self.memory = None

    def identify_invariances(self, source_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        invs = self.invariance_discovery.identify(source_data)
        self.invariances = invs
        return [dict(type=i.type, quantity=i.quantity, tolerance=i.tolerance) for i in invs]

    def _extract_feature_pairs(self, data: List[Dict[str, Any]], max_n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        src: List[np.ndarray] = []
        tgt: List[np.ndarray] = []
        for ex in data[:max_n]:
            s = ex.get('source_state', None)
            t = ex.get('target_state', None)
            if s is None or t is None:
                continue
            src.append(np.asarray(s, dtype=float).flatten())
            tgt.append(np.asarray(t, dtype=float).flatten())
        if not src or not tgt:
            return np.zeros((0, 0)), np.zeros((0, 0))
        xs = np.stack(src, axis=0)
        yt = np.stack(tgt, axis=0)
        d = min(xs.shape[1], yt.shape[1])
        xs = xs[:, :d]
        yt = yt[:, :d]
        return xs, yt

    def learn_domain_mapping(self, key: str, paired_data: List[Dict[str, Any]]) -> bool:
        xs, yt = self._extract_feature_pairs(paired_data)
        if xs.size == 0:
            return False
        if bool(self.config.kernel_mapping):
            mapping2 = NonLinearDomainMapping(kernel='rbf', ridge=self.config.mapping_ridge, gamma=self.config.kernel_gamma)
            ok2 = mapping2.fit(xs, yt)
            if ok2:
                self.domain_mappings[key] = mapping2  # type: ignore[assignment]
            return bool(ok2)

        mapping = DomainMapping(ridge=self.config.mapping_ridge)
        ok = mapping.fit(xs, yt)
        if ok:
            self.domain_mappings[key] = mapping
        return ok

    def train_domain_adapter(self, key: str, paired_data: List[Dict[str, Any]],
                             hidden: Optional[Sequence[int]] = None,
                             steps: int = 200,
                             lr: float = 1e-3) -> bool:
        """Train a DomainAdapter on paired examples (source_state -> target_state).

        This provides a robust trainable path for transfer when closed-form
        mappings are insufficient.
        """
        xs, yt = self._extract_feature_pairs(paired_data, max_n=max(10, len(paired_data)))
        if xs.size == 0:
            return False
        in_dim = int(xs.shape[1])
        out_dim = int(yt.shape[1])
        h = tuple(int(v) for v in (hidden if hidden is not None else (max(16, in_dim * 2), max(16, out_dim * 2))))

        adapter = self.domain_adapters.get(key)
        if adapter is None or adapter.in_dim != in_dim or adapter.out_dim != out_dim:
            adapter = DomainAdapter(in_dim=in_dim, hidden_dims=h, out_dim=out_dim, label=f'adapter_{key}')
            self.domain_adapters[key] = adapter

        n = int(xs.shape[0])
        steps = int(max(1, steps))
        for _ in range(steps):
            idx = int(self._rng.randint(0, n))
            x = Tensor(xs[idx])
            y = Tensor(yt[idx])
            pred = adapter(x)
            err = pred - y
            loss = (err * err).sum()
            for p in adapter.parameters():
                p.grad = np.zeros_like(p.data)
            loss.backward()
            _sgd_step(adapter.parameters(), lr=float(lr), grad_clip=10.0)

        return True

    def apply_domain_adapter(self, key: str, x: np.ndarray) -> np.ndarray:
        adapter = self.domain_adapters.get(key)
        if adapter is None:
            return np.asarray(x, dtype=float)
        x = np.asarray(x, dtype=float).flatten()
        out = adapter(Tensor(x)).data
        return np.asarray(out, dtype=float)

    def apply_domain_mapping(self, key: str, x: np.ndarray) -> np.ndarray:
        mapping = self.domain_mappings.get(key, None)
        if mapping is None:
            return np.asarray(x, dtype=float)
        return mapping.apply(x)

    def meta_learn_adaptation(self, task_batch: List[Dict[str, Any]]) -> None:
        """Train meta-learner if available. task_batch: [{'support': [...], 'query': [...]}]."""
        if self.meta_learner is None:
            return
        try:
            self.meta_learner.train_step(task_batch)
        except Exception:
            return

    def fast_adapt(self, task_id: str, support_examples: List[Dict[str, Any]], num_steps: int = 5) -> Optional[np.ndarray]:
        if self.meta_learner is None:
            return None
        try:
            return self.meta_learner.fast_adapt(task_id=task_id, support_examples=support_examples, num_steps=int(num_steps))
        except Exception:
            return None

    def apply_invariances(self, target_world_model: Any) -> None:
        for inv in self.invariances:
            if inv.type == 'conservation' and inv.quantity == 'energy':
                try:
                    if hasattr(target_world_model, 'physics') and hasattr(target_world_model.physics, 'conservation_enabled'):
                        target_world_model.physics.conservation_enabled = True
                except Exception:
                    pass

    def transfer_to_target(self, target_data: List[Dict[str, Any]], target_world_model: Any) -> Dict[str, Any]:
        self.apply_invariances(target_world_model)
        mapping_ok = self.learn_domain_mapping('source_to_target', target_data)
        adapter_ok = self.train_domain_adapter('source_to_target', target_data, steps=100, lr=1e-3) if target_data else False
        return {
            'num_invariances_transferred': int(len(self.invariances)),
            'mapping_learned': bool(mapping_ok),
            'adapter_trained': bool(adapter_ok),
        }

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        try:
            params.extend(self.progressive.parameters())
        except Exception:
            pass
        for ad in self.domain_adapters.values():
            try:
                params.extend(ad.parameters())
            except Exception:
                pass
        if self.meta_learner is not None and hasattr(self.meta_learner, 'parameters'):
            try:
                params.extend(self.meta_learner.parameters())
            except Exception:
                pass
        return params


class TransferLearningEngine:
    def __init__(self, source_world_model: Any, config: Optional[KnowledgeTransferConfig] = None):
        self.system = AGIKnowledgeTransferSystem(source_world_model=source_world_model, config=config)

    @property
    def source_model(self) -> Any:
        return self.system.source_model

    @property
    def invariances(self) -> List[Dict[str, Any]]:
        return [dict(type=i.type, quantity=i.quantity, tolerance=i.tolerance) for i in self.system.invariances]

    @property
    def domain_mappings(self) -> Dict[str, Any]:
        return {k: v.mapping for k, v in self.system.domain_mappings.items()}

    def identify_invariances(self, source_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.system.identify_invariances(source_data)

    def transfer_to_target(self, target_data: List[Dict[str, Any]], target_world_model: Any) -> Dict[str, Any]:
        return self.system.transfer_to_target(target_data=target_data, target_world_model=target_world_model)

    def train_domain_adapter(self, key: str, paired_data: List[Dict[str, Any]],
                             hidden: Optional[Sequence[int]] = None,
                             steps: int = 200,
                             lr: float = 1e-3) -> bool:
        return self.system.train_domain_adapter(key=key, paired_data=paired_data, hidden=hidden, steps=steps, lr=lr)

    def apply_domain_adapter(self, key: str, x: np.ndarray) -> np.ndarray:
        return self.system.apply_domain_adapter(key=key, x=x)

    def meta_learn_adaptation(self, task_batch: List[Dict[str, Any]]) -> None:
        self.system.meta_learn_adaptation(task_batch=task_batch)

    def fast_adapt(self, task_id: str, support_examples: List[Dict[str, Any]], num_steps: int = 5) -> Optional[np.ndarray]:
        return self.system.fast_adapt(task_id=task_id, support_examples=support_examples, num_steps=num_steps)

    def parameters(self) -> List[Tensor]:
        return self.system.parameters()
