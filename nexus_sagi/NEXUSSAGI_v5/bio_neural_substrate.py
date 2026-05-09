import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import scipy.sparse as _sp
    import scipy.sparse.csgraph as _csg
except Exception:
    _sp = None
    _csg = None


# =============================================================================
# PHASE I: FOUNDATIONAL NEURAL PRIMITIVES
# =============================================================================


@dataclass
class SpikeEvent:
    t: float
    idx: int


class NeuromodulatorState:
    def __init__(
        self,
        dopamine: float = 0.0,
        acetylcholine: float = 0.0,
        norepinephrine: float = 0.0,
        plasticity_gate: float = 1.0,
    ):
        self.dopamine = float(dopamine)
        self.acetylcholine = float(acetylcholine)
        self.norepinephrine = float(norepinephrine)
        self.plasticity_gate = float(plasticity_gate)


class NeuralSubstrate:
    def __init__(
        self,
        evidence_dim: int,
        latent_dim: int,
        label: str = "neural_substrate",
        dt_ms: float = 1.0,
        seed: int = 0,
    ):
        self.evidence_dim = int(evidence_dim)
        self.latent_dim = int(latent_dim)
        self.label = str(label)
        self.dt_ms = float(dt_ms)
        self._rng = np.random.RandomState(int(seed))

        self.column = CanonicalCorticalColumn(dt_ms=float(self.dt_ms), rng=self._rng)
        self.column.reset_state()

        self._t_ms = 0.0

        # Per-synapse eligibility traces for reward/neuromodulator-gated plasticity.
        # Shape convention matches ConductanceSynapses.w: (n_post, n_pre)
        self._plasticity_traces: Dict[str, Dict[str, Any]] = {}
        for syn_name, syn in [
            ('L23L23', self.column.syn_l23_l23),
            ('L4L23', self.column.syn_l4_l23),
            ('L5L23', self.column.syn_l5_l23),
            ('L23L4', self.column.syn_l23_l4),
            ('L6L4', self.column.syn_l6_l4),
            ('L23L5', self.column.syn_l23_l5),
            ('L4L5', self.column.syn_l4_l5),
            ('L5L5', self.column.syn_l5_l5),
            ('L6L5', self.column.syn_l6_l5),
            ('L6L6', self.column.syn_l6_l6),
            ('L5L6', self.column.syn_l5_l6),
        ]:
            self._plasticity_traces[syn_name] = {
                'pre_trace': np.zeros((syn.n_pre,), dtype=np.float64),
                'post_trace': np.zeros((syn.n_post,), dtype=np.float64),
                'last_t_ms': 0.0,
            }

        self._last_latent = np.zeros((self.latent_dim,), dtype=np.float64)
        self._last_key = np.zeros((self.latent_dim,), dtype=np.float64)

        self._key_proj = self._rng.randn(self.latent_dim, self.latent_dim).astype(np.float64) * (1.0 / math.sqrt(max(1, self.latent_dim)))

    def reset_state(self) -> None:
        self.column.reset_state()
        self._last_latent[...] = 0.0
        self._last_key[...] = 0.0

        self._t_ms = 0.0
        for tr in self._plasticity_traces.values():
            tr['pre_trace'][...] = 0.0
            tr['post_trace'][...] = 0.0
            tr['last_t_ms'] = 0.0

    def forward(
        self,
        evidence: np.ndarray,
        modulators: Optional[NeuromodulatorState] = None,
        context: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        e = np.asarray(evidence, dtype=np.float64).reshape(-1)
        aligned = np.zeros((self.column.cfg.n_l4,), dtype=np.float64)
        k = min(aligned.size, e.size)
        if k > 0:
            aligned[:k] = e[:k]

        gain = 1.0
        if modulators is not None:
            gain = 1.0 + 0.25 * np.tanh(float(modulators.acetylcholine)) + 0.25 * np.tanh(float(modulators.norepinephrine))

        context_gain = 0.0
        if context is not None:
            ctx = np.asarray(context, dtype=np.float64).reshape(-1)
            if ctx.size > 0:
                context_gain = float(np.tanh(np.mean(ctx)))

        aligned = aligned * float(gain) + (0.10 * context_gain)

        sp = self.column.step(aligned)

        self._t_ms += float(self.dt_ms)

        feat = np.concatenate(
            [
                np.mean(sp['L4']).reshape(1),
                np.mean(sp['L23']).reshape(1),
                np.mean(sp['L5']).reshape(1),
                np.mean(sp['L6']).reshape(1),
                np.asarray(self.column.l23.v, dtype=np.float64).reshape(-1),
                np.asarray(self.column.l4.v, dtype=np.float64).reshape(-1),
                np.asarray(self.column.l5.v, dtype=np.float64).reshape(-1),
                np.asarray(self.column.l6.v, dtype=np.float64).reshape(-1),
            ],
            axis=0,
        )
        if feat.size < self.latent_dim:
            feat = np.pad(feat, (0, self.latent_dim - feat.size))
        latent = feat[: self.latent_dim].copy()
        latent = np.tanh(latent)

        key = np.matmul(self._key_proj, latent)
        key = np.tanh(key)

        self._last_latent = latent.copy()
        self._last_key = key.copy()

        stats: Dict[str, Any] = {
            'gain': float(gain),
            'context_gain': float(context_gain),
            'l4_rate': float(np.mean(sp['L4'])),
            'l23_rate': float(np.mean(sp['L23'])),
            'l5_rate': float(np.mean(sp['L5'])),
            'l6_rate': float(np.mean(sp['L6'])),
            'stp_enabled': True,
        }
        return latent, stats

    def step_plasticity(
        self,
        modulators: NeuromodulatorState,
        reward: Optional[float] = None,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        g = float(max(0.0, getattr(modulators, 'plasticity_gate', 1.0)))
        if g <= 0.0:
            return {'plasticity_applied': False}

        r = 0.0 if reward is None else float(reward)
        da = float(getattr(modulators, 'dopamine', 0.0))
        ach = float(getattr(modulators, 'acetylcholine', 0.0))
        ne = float(getattr(modulators, 'norepinephrine', 0.0))

        # Neuromodulator-gated learning rate:
        # - Dopamine + reward increase update magnitude.
        # - ACh increases plasticity (learning rate / attention).
        # - NE reduces plasticity under high arousal to keep stability.
        mod_scale = (0.5 + 0.5 * float(np.tanh(da + 0.1 * r)))
        ach_gain = 0.75 + 0.5 * float(np.tanh(ach))
        ne_damp = 1.0 / (1.0 + 0.5 * float(np.tanh(max(0.0, ne))))
        scale = float(lr) * g * mod_scale * ach_gain * ne_damp

        # STDP trace parameters (kept stable, ms scale)
        tau_pre_ms = 20.0
        tau_post_ms = 40.0
        a_plus = 0.01
        a_minus = 0.012

        t = float(self._t_ms)

        def _align_spikes(x: np.ndarray, n: int) -> np.ndarray:
            s = np.asarray(x, dtype=np.float64).reshape(-1)
            if s.size != int(n):
                aligned = np.zeros((int(n),), dtype=np.float64)
                k = min(int(n), int(s.size))
                if k > 0:
                    aligned[:k] = s[:k]
                s = aligned
            return (s > 0.0).astype(np.float64)

        syn_specs = [
            ('L23L23', 'L23', 'L23', self.column.syn_l23_l23),
            ('L4L23', 'L4', 'L23', self.column.syn_l4_l23),
            ('L5L23', 'L5', 'L23', self.column.syn_l5_l23),
            ('L23L4', 'L23', 'L4', self.column.syn_l23_l4),
            ('L6L4', 'L6', 'L4', self.column.syn_l6_l4),
            ('L23L5', 'L23', 'L5', self.column.syn_l23_l5),
            ('L4L5', 'L4', 'L5', self.column.syn_l4_l5),
            ('L5L5', 'L5', 'L5', self.column.syn_l5_l5),
            ('L6L5', 'L6', 'L5', self.column.syn_l6_l5),
            ('L6L6', 'L6', 'L6', self.column.syn_l6_l6),
            ('L5L6', 'L5', 'L6', self.column.syn_l5_l6),
        ]

        applied = False
        try:
            for name, pre_layer, post_layer, syn in syn_specs:
                tr = self._plasticity_traces.get(name)
                if tr is None:
                    continue

                dt = float(max(0.0, t - float(tr.get('last_t_ms', 0.0))))
                tr['last_t_ms'] = t
                if dt > 0.0:
                    tr['pre_trace'] = tr['pre_trace'] * math.exp(-dt / tau_pre_ms)
                    tr['post_trace'] = tr['post_trace'] * math.exp(-dt / tau_post_ms)

                s_pre = _align_spikes(self.column._last_spikes.get(pre_layer, np.zeros((syn.n_pre,))), syn.n_pre)
                s_post = _align_spikes(self.column._last_spikes.get(post_layer, np.zeros((syn.n_post,))), syn.n_post)

                # Eligibility increment
                if np.any(s_post > 0.0):
                    dw = scale * a_plus * (s_post.reshape(-1, 1) * tr['pre_trace'].reshape(1, -1))
                    syn.w = syn.w + dw
                if np.any(s_pre > 0.0):
                    dw = -scale * a_minus * (tr['post_trace'].reshape(-1, 1) * s_pre.reshape(1, -1))
                    syn.w = syn.w + dw

                tr['pre_trace'] = tr['pre_trace'] + s_pre
                tr['post_trace'] = tr['post_trace'] + s_post

                # Stability: clip weights
                syn.w = np.clip(syn.w, -1.0, 1.0)
                applied = True
        except Exception:
            applied = False

        return {
            'plasticity_applied': bool(applied),
            'plasticity_scale': float(scale),
            'reward': float(r),
            'da': float(da),
        }

    @property
    def last_latent(self) -> np.ndarray:
        return self._last_latent.copy()

    @property
    def last_key(self) -> np.ndarray:
        return self._last_key.copy()


class LIFPopulation:
    def __init__(
        self,
        n: int,
        tau_m: float = 10.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_th: float = -55.0,
        r_m: float = 1.0,
        refractory_ms: float = 2.0,
        label: str = "lif",
    ):
        self.n = int(n)
        self.tau_m = float(max(1e-9, tau_m))
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_th = float(v_th)
        self.r_m = float(r_m)
        self.refractory_ms = float(max(0.0, refractory_ms))
        self.label = str(label)

        self.v = np.full((self.n,), self.v_rest, dtype=np.float64)
        self.refrac_left = np.zeros((self.n,), dtype=np.float64)

    def reset_state(self) -> None:
        self.v[...] = self.v_rest
        self.refrac_left[...] = 0.0

    def step(self, i_in: np.ndarray, dt_ms: float) -> np.ndarray:
        dt = float(dt_ms)
        x = np.asarray(i_in, dtype=np.float64).reshape(-1)
        if x.size != self.n:
            aligned = np.zeros((self.n,), dtype=np.float64)
            k = min(self.n, x.size)
            if k > 0:
                aligned[:k] = x[:k]
            x = aligned

        active = self.refrac_left <= 0.0

        dv = np.zeros_like(self.v)
        dv[active] = (-(self.v[active] - self.v_rest) + self.r_m * x[active]) * (dt / self.tau_m)
        self.v = self.v + dv

        spikes = (self.v >= self.v_th) & active
        if np.any(spikes):
            self.v[spikes] = self.v_reset
            self.refrac_left[spikes] = self.refractory_ms

        self.refrac_left = np.maximum(0.0, self.refrac_left - dt)
        return spikes.astype(np.float64)


class IzhikevichPopulation:
    def __init__(
        self,
        n: int,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        v_init: float = -70.0,
        label: str = "izh",
    ):
        self.n = int(n)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.label = str(label)

        self.v = np.full((self.n,), float(v_init), dtype=np.float64)
        self.u = self.b * self.v

    def reset_state(self) -> None:
        self.v[...] = -70.0
        self.u[...] = self.b * self.v

    def step(self, i_in: np.ndarray, dt_ms: float = 1.0) -> np.ndarray:
        dt = float(dt_ms)
        x = np.asarray(i_in, dtype=np.float64).reshape(-1)
        if x.size != self.n:
            aligned = np.zeros((self.n,), dtype=np.float64)
            k = min(self.n, x.size)
            if k > 0:
                aligned[:k] = x[:k]
            x = aligned

        dv = (0.04 * (self.v ** 2) + 5.0 * self.v + 140.0 - self.u + x) * dt
        du = (self.a * (self.b * self.v - self.u)) * dt

        self.v = self.v + dv
        self.u = self.u + du

        spikes = self.v >= 30.0
        if np.any(spikes):
            self.v[spikes] = self.c
            self.u[spikes] = self.u[spikes] + self.d
        return spikes.astype(np.float64)


class MultiCompartmentNeuron:
    def __init__(
        self,
        tau_a: float = 2.0,
        tau_b: float = 2.0,
        tau_l: float = 4.0,
        g_a: float = 0.5,
        g_b: float = 0.5,
        g_l: float = 1.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
    ):
        self.tau_a = float(max(1e-9, tau_a))
        self.tau_b = float(max(1e-9, tau_b))
        self.tau_l = float(max(1e-9, tau_l))
        self.g_a = float(g_a)
        self.g_b = float(g_b)
        self.g_l = float(g_l)
        self.v_th = float(v_th)
        self.v_reset = float(v_reset)

        self.u_a = 0.0
        self.u_b = 0.0
        self.u_l = 0.0

    def reset_state(self) -> None:
        self.u_a = 0.0
        self.u_b = 0.0
        self.u_l = 0.0

    def step(self, x_basal: float, x_apical: float, dt_ms: float = 1.0) -> float:
        dt = float(dt_ms)
        self.u_a = self.u_a + dt * (-(self.u_a - float(x_apical)) / self.tau_a)
        self.u_b = self.u_b + dt * (-(self.u_b - float(x_basal)) / self.tau_b)

        v_soma = (self.g_a * self.u_a) + (self.g_b * self.u_b) + (self.g_l * self.u_l)
        self.u_l = self.u_l + dt * (-(self.u_l - v_soma) / self.tau_l)

        spike = 1.0 if self.u_l > self.v_th else 0.0
        if spike > 0.0:
            self.u_l = self.v_reset
        return spike


class SRMPopulation:
    def __init__(
        self,
        n: int,
        tau_eps: float = 10.0,
        tau_eta: float = 20.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        refractory_ms: float = 2.0,
        label: str = "srm",
    ):
        self.n = int(n)
        self.tau_eps = float(max(1e-9, tau_eps))
        self.tau_eta = float(max(1e-9, tau_eta))
        self.v_th = float(v_th)
        self.v_reset = float(v_reset)
        self.refractory_ms = float(max(0.0, refractory_ms))
        self.label = str(label)

        self.v = np.zeros((self.n,), dtype=np.float64)
        self.eps_trace = np.zeros((self.n,), dtype=np.float64)
        self.eta_trace = np.zeros((self.n,), dtype=np.float64)
        self.refrac_left = np.zeros((self.n,), dtype=np.float64)

    def reset_state(self) -> None:
        self.v[...] = 0.0
        self.eps_trace[...] = 0.0
        self.eta_trace[...] = 0.0
        self.refrac_left[...] = 0.0

    def step(self, weighted_spike_input: np.ndarray, dt_ms: float) -> np.ndarray:
        dt = float(dt_ms)
        x = np.asarray(weighted_spike_input, dtype=np.float64).reshape(-1)
        if x.size != self.n:
            aligned = np.zeros((self.n,), dtype=np.float64)
            k = min(self.n, x.size)
            if k > 0:
                aligned[:k] = x[:k]
            x = aligned

        self.eps_trace = self.eps_trace * math.exp(-dt / self.tau_eps) + x
        self.eta_trace = self.eta_trace * math.exp(-dt / self.tau_eta)

        active = self.refrac_left <= 0.0
        self.v = self.eps_trace + self.eta_trace

        spikes = (self.v >= self.v_th) & active
        if np.any(spikes):
            self.v[spikes] = self.v_reset
            self.eta_trace[spikes] = -1.0
            self.refrac_left[spikes] = self.refractory_ms

        self.refrac_left = np.maximum(0.0, self.refrac_left - dt)
        return spikes.astype(np.float64)


class HodgkinHuxleyPopulation:
    def __init__(
        self,
        n: int,
        c_m: float = 1.0,
        g_na: float = 120.0,
        g_k: float = 36.0,
        g_l: float = 0.3,
        e_na: float = 50.0,
        e_k: float = -77.0,
        e_l: float = -54.387,
        label: str = "hh",
    ):
        self.n = int(n)
        self.c_m = float(c_m)
        self.g_na = float(g_na)
        self.g_k = float(g_k)
        self.g_l = float(g_l)
        self.e_na = float(e_na)
        self.e_k = float(e_k)
        self.e_l = float(e_l)
        self.label = str(label)

        self.v = np.full((self.n,), -65.0, dtype=np.float64)
        self.m = np.full((self.n,), 0.05, dtype=np.float64)
        self.h = np.full((self.n,), 0.6, dtype=np.float64)
        self.n_gate = np.full((self.n,), 0.32, dtype=np.float64)

    def reset_state(self) -> None:
        self.v[...] = -65.0
        self.m[...] = 0.05
        self.h[...] = 0.6
        self.n_gate[...] = 0.32

    def _alpha_m(self, v: np.ndarray) -> np.ndarray:
        return (0.1 * (v + 40.0)) / (1.0 - np.exp(-(v + 40.0) / 10.0) + 1e-12)

    def _beta_m(self, v: np.ndarray) -> np.ndarray:
        return 4.0 * np.exp(-(v + 65.0) / 18.0)

    def _alpha_h(self, v: np.ndarray) -> np.ndarray:
        return 0.07 * np.exp(-(v + 65.0) / 20.0)

    def _beta_h(self, v: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))

    def _alpha_n(self, v: np.ndarray) -> np.ndarray:
        return (0.01 * (v + 55.0)) / (1.0 - np.exp(-(v + 55.0) / 10.0) + 1e-12)

    def _beta_n(self, v: np.ndarray) -> np.ndarray:
        return 0.125 * np.exp(-(v + 65.0) / 80.0)

    def step(self, i_ext: np.ndarray, dt_ms: float) -> Tuple[np.ndarray, np.ndarray]:
        dt = float(dt_ms)
        i = np.asarray(i_ext, dtype=np.float64).reshape(-1)
        if i.size != self.n:
            aligned = np.zeros((self.n,), dtype=np.float64)
            k = min(self.n, i.size)
            if k > 0:
                aligned[:k] = i[:k]
            i = aligned

        v = self.v

        am = self._alpha_m(v)
        bm = self._beta_m(v)
        ah = self._alpha_h(v)
        bh = self._beta_h(v)
        an = self._alpha_n(v)
        bn = self._beta_n(v)

        self.m = self.m + dt * (am * (1.0 - self.m) - bm * self.m)
        self.h = self.h + dt * (ah * (1.0 - self.h) - bh * self.h)
        self.n_gate = self.n_gate + dt * (an * (1.0 - self.n_gate) - bn * self.n_gate)

        i_na = self.g_na * (self.m ** 3) * self.h * (v - self.e_na)
        i_k = self.g_k * (self.n_gate ** 4) * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)

        dv = (i - i_na - i_k - i_l) / self.c_m
        self.v = self.v + dt * dv

        spikes = (self.v >= 0.0).astype(np.float64)
        return spikes, self.v.copy()


# =============================================================================
# PHASE I: SYNAPTIC TRANSMISSION (CONDUCTANCE-BASED)
# =============================================================================


class ConductanceSynapses:
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_syn_ms: float,
        e_syn_mV: float,
        g_max: float,
        delays_ms: float = 1.0,
        label: str = "syn",
    ):
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self.tau_syn_ms = float(max(1e-9, tau_syn_ms))
        self.e_syn_mV = float(e_syn_mV)
        self.g_max = float(g_max)
        self.delays_ms = float(max(0.0, delays_ms))
        self.label = str(label)

        self.w = np.random.randn(self.n_post, self.n_pre).astype(np.float64) * 0.01
        self.g = np.zeros((self.n_post, self.n_pre), dtype=np.float64)

        self.use_stp = False
        self.u_stp = np.full((self.n_post, self.n_pre), 0.2, dtype=np.float64)
        self.r_stp = np.ones((self.n_post, self.n_pre), dtype=np.float64)
        self.tau_d_ms = 200.0
        self.tau_f_ms = 150.0
        self.u_base = 0.2

        self._delay_steps = 0
        self._spike_buffer: List[np.ndarray] = []

    def set_dt(self, dt_ms: float) -> None:
        self._delay_steps = int(round(self.delays_ms / float(max(1e-9, dt_ms))))
        self._spike_buffer = [np.zeros((self.n_pre,), dtype=np.float64) for _ in range(max(1, self._delay_steps + 1))]

    def reset_state(self) -> None:
        self.g[...] = 0.0
        self.u_stp[...] = float(self.u_base)
        self.r_stp[...] = 1.0
        for i in range(len(self._spike_buffer)):
            self._spike_buffer[i][...] = 0.0

    def configure_stp(self, enabled: bool, u_base: float = 0.2, tau_d_ms: float = 200.0, tau_f_ms: float = 150.0) -> None:
        self.use_stp = bool(enabled)
        self.u_base = float(u_base)
        self.tau_d_ms = float(max(1e-9, tau_d_ms))
        self.tau_f_ms = float(max(1e-9, tau_f_ms))
        self.u_stp[...] = float(self.u_base)
        self.r_stp[...] = 1.0

    def step(self, spikes_pre: np.ndarray, v_post: np.ndarray, dt_ms: float) -> np.ndarray:
        dt = float(dt_ms)
        s = np.asarray(spikes_pre, dtype=np.float64).reshape(-1)
        if s.size != self.n_pre:
            aligned = np.zeros((self.n_pre,), dtype=np.float64)
            k = min(self.n_pre, s.size)
            if k > 0:
                aligned[:k] = s[:k]
            s = aligned

        if not self._spike_buffer:
            self.set_dt(dt_ms)

        self._spike_buffer.append(s.copy())
        delayed = self._spike_buffer.pop(0)

        decay = math.exp(-dt / self.tau_syn_ms)
        self.g = self.g * decay

        if self.use_stp:
            self.u_stp = self.u_stp + dt * ((self.u_base - self.u_stp) / self.tau_f_ms)
            self.r_stp = self.r_stp + dt * ((1.0 - self.r_stp) / self.tau_d_ms)
            spike_mask = (delayed.reshape(1, -1) > 0.0).astype(np.float64)
            self.u_stp = self.u_stp + spike_mask * (1.0 - self.u_stp) * self.u_base
            eff = self.u_stp * self.r_stp * spike_mask
            self.r_stp = self.r_stp - eff
            inc = self.g_max * (eff * self.w)
        else:
            inc = self.g_max * (delayed.reshape(1, -1) * self.w)
        self.g = self.g + inc

        v = np.asarray(v_post, dtype=np.float64).reshape(-1)
        if v.size != self.n_post:
            aligned_v = np.zeros((self.n_post,), dtype=np.float64)
            k = min(self.n_post, v.size)
            if k > 0:
                aligned_v[:k] = v[:k]
            v = aligned_v

        i_syn = np.sum(self.g * (v.reshape(-1, 1) - self.e_syn_mV), axis=1)
        return i_syn


# =============================================================================
# PHASE II: PLASTICITY RULES
# =============================================================================


@dataclass
class PairSTDPParams:
    a_plus: float = 0.01
    a_minus: float = 0.012
    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 40.0


class PairSTDP:
    def __init__(self, n_pre: int, n_post: int, params: Optional[PairSTDPParams] = None):
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self.p = params or PairSTDPParams()
        self.pre_trace = np.zeros((self.n_pre,), dtype=np.float64)
        self.post_trace = np.zeros((self.n_post,), dtype=np.float64)
        self._last_t_ms = 0.0

    def step(self, dt_ms: float) -> None:
        dt = float(dt_ms)
        self.pre_trace = self.pre_trace * math.exp(-dt / self.p.tau_plus_ms)
        self.post_trace = self.post_trace * math.exp(-dt / self.p.tau_minus_ms)

    def update(self, w: np.ndarray, s_pre: np.ndarray, s_post: np.ndarray, t_ms: float) -> np.ndarray:
        t = float(t_ms)
        dt = float(max(0.0, t - float(self._last_t_ms)))
        if dt > 0.0:
            self.step(dt_ms=dt)
        self._last_t_ms = t

        s_pre = np.asarray(s_pre, dtype=np.float64).reshape(-1)
        s_post = np.asarray(s_post, dtype=np.float64).reshape(-1)
        if s_pre.size != self.n_pre:
            aligned = np.zeros((self.n_pre,), dtype=np.float64)
            k = min(self.n_pre, s_pre.size)
            if k > 0:
                aligned[:k] = s_pre[:k]
            s_pre = aligned
        if s_post.size != self.n_post:
            aligned = np.zeros((self.n_post,), dtype=np.float64)
            k = min(self.n_post, s_post.size)
            if k > 0:
                aligned[:k] = s_post[:k]
            s_post = aligned

        pre = (s_pre > 0.0).astype(np.float64)
        post = (s_post > 0.0).astype(np.float64)

        if np.any(post):
            dw = self.p.a_plus * (post.reshape(-1, 1) * self.pre_trace.reshape(1, -1))
            w = w + dw

        if np.any(pre):
            dw = -self.p.a_minus * (self.post_trace.reshape(-1, 1) * pre.reshape(1, -1))
            w = w + dw

        self.pre_trace = self.pre_trace + pre
        self.post_trace = self.post_trace + post
        return w


@dataclass
class TripletSTDPParams:
    tau_x_ms: float = 20.0
    tau_y1_ms: float = 20.0
    tau_y2_ms: float = 40.0
    a_ltp: float = 1e-3
    a_ltd: float = 1e-3


class TripletSTDP:
    def __init__(self, n_pre: int, n_post: int, params: Optional[TripletSTDPParams] = None):
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self.p = params or TripletSTDPParams()
        self.x = np.zeros((self.n_pre,), dtype=np.float64)
        self.y1 = np.zeros((self.n_post,), dtype=np.float64)
        self.y2 = np.zeros((self.n_post,), dtype=np.float64)

    def step(self, dt_ms: float) -> None:
        dt = float(dt_ms)
        self.x = self.x * math.exp(-dt / self.p.tau_x_ms)
        self.y1 = self.y1 * math.exp(-dt / self.p.tau_y1_ms)
        self.y2 = self.y2 * math.exp(-dt / self.p.tau_y2_ms)

    def update(self, w: np.ndarray, s_pre: np.ndarray, s_post: np.ndarray) -> np.ndarray:
        s_pre = np.asarray(s_pre, dtype=np.float64).reshape(-1)
        s_post = np.asarray(s_post, dtype=np.float64).reshape(-1)

        pre = (s_pre > 0.0).astype(np.float64)
        post = (s_post > 0.0).astype(np.float64)

        self.x = self.x + pre
        self.y1 = self.y1 + post
        self.y2 = self.y2 + post

        dw_ltp = self.p.a_ltp * (post.reshape(-1, 1) * (self.x.reshape(1, -1) * self.y2.reshape(-1, 1)))
        dw_ltd = -self.p.a_ltd * (self.y1.reshape(-1, 1) * pre.reshape(1, -1))

        return w + dw_ltp + dw_ltd


@dataclass
class ClopathParams:
    a_ltd: float = 1e-5
    a_ltp: float = 1e-5
    theta_minus: float = -70.0
    theta_plus: float = -55.0
    tau_u_minus_ms: float = 20.0
    tau_u_plus_ms: float = 5.0
    tau_x_ms: float = 20.0


class ClopathPlasticity:
    def __init__(self, n_pre: int, n_post: int, params: Optional[ClopathParams] = None):
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self.p = params or ClopathParams()
        self.u_minus = np.full((self.n_post,), self.p.theta_minus, dtype=np.float64)
        self.u_plus = np.full((self.n_post,), self.p.theta_minus, dtype=np.float64)
        self.x_bar = np.zeros((self.n_pre,), dtype=np.float64)

    def step(self, u_post: np.ndarray, s_pre: np.ndarray, dt_ms: float) -> None:
        dt = float(dt_ms)
        u = np.asarray(u_post, dtype=np.float64).reshape(-1)
        if u.size != self.n_post:
            aligned = np.full((self.n_post,), self.p.theta_minus, dtype=np.float64)
            k = min(self.n_post, u.size)
            if k > 0:
                aligned[:k] = u[:k]
            u = aligned

        self.u_minus = self.u_minus + dt * (-(self.u_minus - u) / self.p.tau_u_minus_ms)
        self.u_plus = self.u_plus + dt * (-(self.u_plus - u) / self.p.tau_u_plus_ms)

        s = np.asarray(s_pre, dtype=np.float64).reshape(-1)
        if s.size != self.n_pre:
            aligned_s = np.zeros((self.n_pre,), dtype=np.float64)
            k = min(self.n_pre, s.size)
            if k > 0:
                aligned_s[:k] = s[:k]
            s = aligned_s
        self.x_bar = self.x_bar * math.exp(-dt / self.p.tau_x_ms) + (s > 0.0).astype(np.float64)

    def update(self, w: np.ndarray, s_pre: np.ndarray, u_post: np.ndarray) -> np.ndarray:
        s_pre = np.asarray(s_pre, dtype=np.float64).reshape(-1)
        u = np.asarray(u_post, dtype=np.float64).reshape(-1)
        if u.size != self.n_post:
            aligned = np.full((self.n_post,), self.p.theta_minus, dtype=np.float64)
            k = min(self.n_post, u.size)
            if k > 0:
                aligned[:k] = u[:k]
            u = aligned

        s = (s_pre > 0.0).astype(np.float64).reshape(1, -1)

        ltd_gate = np.maximum(0.0, self.u_minus - self.p.theta_minus).reshape(-1, 1)
        ltp_gate1 = np.maximum(0.0, u - self.p.theta_plus).reshape(-1, 1)
        ltp_gate2 = np.maximum(0.0, self.u_plus - self.p.theta_minus).reshape(-1, 1)

        dw_ltd = -self.p.a_ltd * (s * ltd_gate)
        dw_ltp = self.p.a_ltp * (self.x_bar.reshape(1, -1) * ltp_gate1 * ltp_gate2)

        return w + dw_ltd + dw_ltp


def oja_update(w: np.ndarray, nu_i: np.ndarray, nu_j: np.ndarray, gamma: float, dt_ms: float) -> np.ndarray:
    dt = float(dt_ms)
    nu_i = np.asarray(nu_i, dtype=np.float64).reshape(-1, 1)
    nu_j = np.asarray(nu_j, dtype=np.float64).reshape(1, -1)
    dw = float(gamma) * (nu_i * nu_j - w * (nu_i ** 2))
    return w + dt * dw


# =============================================================================
# PHASE IV: TEMPORAL BINDING (KURAMOTO)
# =============================================================================


class KuramotoSynchrony:
    def __init__(self, n: int, k: float = 1.0, label: str = "kuramoto"):
        self.n = int(n)
        self.k = float(k)
        self.label = str(label)
        self.phi = np.random.rand(self.n).astype(np.float64) * 2.0 * np.pi
        self.omega = np.random.randn(self.n).astype(np.float64)

    def step(self, coupling: Optional[np.ndarray], dt: float) -> np.ndarray:
        if coupling is None:
            coupling = np.ones((self.n, self.n), dtype=np.float64) * (self.k / float(max(1, self.n)))
        else:
            coupling = np.asarray(coupling, dtype=np.float64)

        dphi = self.omega.copy()
        for i in range(self.n):
            dphi[i] += np.sum(coupling[i, :] * np.sin(self.phi - self.phi[i]))
        self.phi = self.phi + float(dt) * dphi
        return self.phi.copy()


# =============================================================================
# PHASE III: TOPOLOGY (SMALL-WORLD / SCALE-FREE) + BASIC METRICS
# =============================================================================


def watts_strogatz_adjacency(n: int, k: int, p: float, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    rng = rng or np.random.RandomState(0)
    n = int(n)
    k = int(k)
    p = float(p)
    adj = np.zeros((n, n), dtype=np.float64)

    half = k // 2
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if rng.rand() < p:
                candidates = list(set(range(n)) - {i} - set(np.where(adj[i] > 0)[0].tolist()))
                if not candidates:
                    continue
                new_j = int(rng.choice(candidates))
                adj[i, j] = 0.0
                adj[j, i] = 0.0
                adj[i, new_j] = 1.0
                adj[new_j, i] = 1.0

    return adj


def scale_free_weights(adj: np.ndarray, gamma: float = 2.0, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    rng = rng or np.random.RandomState(0)
    a = (np.asarray(adj, dtype=np.float64) > 0).astype(np.float64)
    n = a.shape[0]

    u = rng.rand(n, n)
    w = (1.0 / np.maximum(u, 1e-12)) ** (1.0 / float(max(1e-6, gamma)))
    w = w * a

    w = (w + w.T) / 2.0
    np.fill_diagonal(w, 0.0)
    w = w / (np.max(np.abs(w)) + 1e-12)
    return w


def clustering_coefficient(adj: np.ndarray) -> float:
    a = (np.asarray(adj, dtype=np.float64) > 0).astype(np.float64)
    n = a.shape[0]
    triangles = 0.0
    triples = 0.0
    for i in range(n):
        neigh = np.where(a[i] > 0)[0]
        k = neigh.size
        if k < 2:
            continue
        triples += k * (k - 1) / 2.0
        sub = a[np.ix_(neigh, neigh)]
        triangles += np.sum(sub) / 2.0
    if triples <= 0:
        return 0.0
    return float((triangles) / triples)


def _bfs_distances(a: np.ndarray, src: int) -> np.ndarray:
    n = a.shape[0]
    dist = np.full((n,), np.inf, dtype=np.float64)
    dist[src] = 0.0
    q = [src]
    while q:
        u = q.pop(0)
        for v in np.where(a[u] > 0)[0]:
            if not np.isfinite(dist[v]):
                dist[v] = dist[u] + 1.0
                q.append(int(v))
    return dist


def characteristic_path_length(adj: np.ndarray) -> float:
    a = (np.asarray(adj, dtype=np.float64) > 0).astype(np.float64)
    n = a.shape[0]

    if _sp is not None and _csg is not None:
        try:
            g = _sp.csr_matrix(a)
            dist = _csg.shortest_path(g, directed=False, unweighted=True)
            mask = np.isfinite(dist) & (dist > 0)
            if np.any(mask):
                return float(np.mean(dist[mask]))
        except Exception:
            pass

    dsum = 0.0
    cnt = 0
    for i in range(n):
        dist = _bfs_distances(a, i)
        for j in range(n):
            if i == j:
                continue
            if np.isfinite(dist[j]):
                dsum += float(dist[j])
                cnt += 1
    if cnt == 0:
        return float("inf")
    return float(dsum / float(cnt))


def small_world_index(adj: np.ndarray, rng: Optional[np.random.RandomState] = None) -> float:
    rng = rng or np.random.RandomState(0)
    a = (np.asarray(adj, dtype=np.float64) > 0).astype(np.float64)
    n = a.shape[0]

    c = clustering_coefficient(a)
    l = characteristic_path_length(a)

    edges = np.argwhere(np.triu(a, 1) > 0)
    m = edges.shape[0]
    rand_adj = np.zeros_like(a)
    if m > 0:
        choices = set()
        while len(choices) < m:
            i = int(rng.randint(0, n))
            j = int(rng.randint(0, n))
            if i == j:
                continue
            if i > j:
                i, j = j, i
            choices.add((i, j))
        for i, j in choices:
            rand_adj[i, j] = 1.0
            rand_adj[j, i] = 1.0

    c_r = clustering_coefficient(rand_adj)
    l_r = characteristic_path_length(rand_adj)

    if c_r <= 0 or not np.isfinite(l) or not np.isfinite(l_r) or l_r <= 0:
        return 0.0

    return float((c / c_r) / (l / l_r))


# =============================================================================
# VALIDATION METRICS: SPIKE TRAINS + AVALANCHES + POWER-LAW FITS
# =============================================================================


def isi_cv(spike_times: List[float]) -> float:
    if len(spike_times) < 3:
        return 0.0
    st = np.asarray(spike_times, dtype=np.float64)
    isi = np.diff(st)
    if isi.size == 0:
        return 0.0
    mu = float(np.mean(isi))
    sd = float(np.std(isi))
    if mu <= 1e-12:
        return 0.0
    return float(sd / mu)


def detect_avalanches(pop_spikes: np.ndarray) -> List[int]:
    s = (np.asarray(pop_spikes, dtype=np.float64) > 0.0).astype(np.int64)
    activity = np.sum(s, axis=1)

    sizes: List[int] = []
    cur = 0
    in_av = False
    for a in activity:
        if a > 0:
            cur += int(a)
            in_av = True
        else:
            if in_av:
                sizes.append(int(cur))
            cur = 0
            in_av = False
    if in_av and cur > 0:
        sizes.append(int(cur))
    return sizes


def fit_power_law_exponent(samples: List[float], xmin: Optional[float] = None) -> float:
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 3:
        return 0.0
    if xmin is None:
        xmin = float(np.min(x))
    x = x[x >= float(xmin)]
    if x.size < 3:
        return 0.0

    alpha = 1.0 + float(x.size) / float(np.sum(np.log(x / float(xmin))))
    return float(alpha)


# =============================================================================
# CANONICAL MICRO-CIRCUIT (CONFIGURABLE SCALE)
# =============================================================================


@dataclass
class ColumnConfig:
    n_l23: int = 80
    n_l4: int = 32
    n_l5: int = 64
    n_l6: int = 64
    inh_ratio: float = 0.25


class CanonicalCorticalColumn:
    def __init__(self, cfg: Optional[ColumnConfig] = None, dt_ms: float = 1.0, rng: Optional[np.random.RandomState] = None):
        self.cfg = cfg or ColumnConfig()
        self.dt_ms = float(dt_ms)
        self.rng = rng or np.random.RandomState(0)

        self.l23 = LIFPopulation(self.cfg.n_l23, label="L23")
        self.l4 = LIFPopulation(self.cfg.n_l4, label="L4")
        self.l5 = LIFPopulation(self.cfg.n_l5, label="L5")
        self.l6 = LIFPopulation(self.cfg.n_l6, label="L6")

        def mk_syn(n_pre: int, n_post: int, tau: float, e: float, g: float, delay: float, name: str) -> ConductanceSynapses:
            syn = ConductanceSynapses(n_pre=n_pre, n_post=n_post, tau_syn_ms=tau, e_syn_mV=e, g_max=g, delays_ms=delay, label=name)
            syn.set_dt(self.dt_ms)
            return syn

        self.syn_l23_l23 = mk_syn(self.cfg.n_l23, self.cfg.n_l23, tau=2.0, e=0.0, g=0.1, delay=1.0, name="L23L23")
        self.syn_l4_l23 = mk_syn(self.cfg.n_l4, self.cfg.n_l23, tau=2.0, e=0.0, g=0.1, delay=1.0, name="L4L23")
        self.syn_l5_l23 = mk_syn(self.cfg.n_l5, self.cfg.n_l23, tau=2.0, e=0.0, g=0.1, delay=1.0, name="L5L23")
        self.syn_l23_l4 = mk_syn(self.cfg.n_l23, self.cfg.n_l4, tau=2.0, e=0.0, g=0.1, delay=1.0, name="L23L4")
        self.syn_l6_l4 = mk_syn(self.cfg.n_l6, self.cfg.n_l4, tau=5.0, e=-70.0, g=0.1, delay=2.0, name="L6L4")
        self.syn_l23_l5 = mk_syn(self.cfg.n_l23, self.cfg.n_l5, tau=2.0, e=0.0, g=0.1, delay=1.0, name="L23L5")
        self.syn_l4_l5 = mk_syn(self.cfg.n_l4, self.cfg.n_l5, tau=2.0, e=0.0, g=0.1, delay=1.0, name="L4L5")
        self.syn_l5_l5 = mk_syn(self.cfg.n_l5, self.cfg.n_l5, tau=2.0, e=0.0, g=0.1, delay=1.0, name="L5L5")
        self.syn_l6_l5 = mk_syn(self.cfg.n_l6, self.cfg.n_l5, tau=5.0, e=-70.0, g=0.1, delay=2.0, name="L6L5")
        self.syn_l6_l6 = mk_syn(self.cfg.n_l6, self.cfg.n_l6, tau=2.0, e=0.0, g=0.1, delay=1.0, name="L6L6")
        self.syn_l5_l6 = mk_syn(self.cfg.n_l5, self.cfg.n_l6, tau=2.0, e=0.0, g=0.1, delay=1.0, name="L5L6")

        for syn in [
            self.syn_l23_l23,
            self.syn_l4_l23,
            self.syn_l5_l23,
            self.syn_l23_l4,
            self.syn_l6_l4,
            self.syn_l23_l5,
            self.syn_l4_l5,
            self.syn_l5_l5,
            self.syn_l6_l5,
            self.syn_l6_l6,
            self.syn_l5_l6,
        ]:
            try:
                syn.configure_stp(enabled=True, u_base=0.2, tau_d_ms=200.0, tau_f_ms=150.0)
            except Exception:
                pass

        self._last_spikes: Dict[str, np.ndarray] = {
            'L23': np.zeros((self.cfg.n_l23,), dtype=np.float64),
            'L4': np.zeros((self.cfg.n_l4,), dtype=np.float64),
            'L5': np.zeros((self.cfg.n_l5,), dtype=np.float64),
            'L6': np.zeros((self.cfg.n_l6,), dtype=np.float64),
        }

    def reset_state(self) -> None:
        self.l23.reset_state()
        self.l4.reset_state()
        self.l5.reset_state()
        self.l6.reset_state()

        for syn in [
            self.syn_l23_l23,
            self.syn_l4_l23,
            self.syn_l5_l23,
            self.syn_l23_l4,
            self.syn_l6_l4,
            self.syn_l23_l5,
            self.syn_l4_l5,
            self.syn_l5_l5,
            self.syn_l6_l5,
            self.syn_l6_l6,
            self.syn_l5_l6,
        ]:
            syn.reset_state()

        for k in self._last_spikes:
            self._last_spikes[k][...] = 0.0

    def step(self, input_l4_current: np.ndarray) -> Dict[str, np.ndarray]:
        dt = self.dt_ms

        i_l23 = (
            self.syn_l23_l23.step(self._last_spikes['L23'], self.l23.v, dt)
            + self.syn_l4_l23.step(self._last_spikes['L4'], self.l23.v, dt)
            + self.syn_l5_l23.step(self._last_spikes['L5'], self.l23.v, dt)
        )
        i_l4 = (
            self.syn_l23_l4.step(self._last_spikes['L23'], self.l4.v, dt)
            + self.syn_l6_l4.step(self._last_spikes['L6'], self.l4.v, dt)
            + np.asarray(input_l4_current, dtype=np.float64).reshape(-1)[: self.cfg.n_l4]
        )
        i_l5 = (
            self.syn_l23_l5.step(self._last_spikes['L23'], self.l5.v, dt)
            + self.syn_l4_l5.step(self._last_spikes['L4'], self.l5.v, dt)
            + self.syn_l5_l5.step(self._last_spikes['L5'], self.l5.v, dt)
            + self.syn_l6_l5.step(self._last_spikes['L6'], self.l5.v, dt)
        )
        i_l6 = self.syn_l6_l6.step(self._last_spikes['L6'], self.l6.v, dt) + self.syn_l5_l6.step(self._last_spikes['L5'], self.l6.v, dt)

        s_l23 = self.l23.step(i_l23, dt)
        s_l4 = self.l4.step(i_l4, dt)
        s_l5 = self.l5.step(i_l5, dt)
        s_l6 = self.l6.step(i_l6, dt)

        self._last_spikes['L23'] = s_l23
        self._last_spikes['L4'] = s_l4
        self._last_spikes['L5'] = s_l5
        self._last_spikes['L6'] = s_l6

        return {
            'L23': s_l23.copy(),
            'L4': s_l4.copy(),
            'L5': s_l5.copy(),
            'L6': s_l6.copy(),
        }


# =============================================================================
# END-TO-END RUNNER (USED BY TESTS)
# =============================================================================


def run_column_simulation(
    steps: int = 200,
    dt_ms: float = 1.0,
    input_scale: float = 5.0,
    seed: int = 0,
) -> Dict[str, Any]:
    rng = np.random.RandomState(int(seed))
    col = CanonicalCorticalColumn(dt_ms=float(dt_ms), rng=rng)
    col.reset_state()

    all_spikes_l23 = []
    all_spikes_l4 = []
    all_spikes_l5 = []
    all_spikes_l6 = []

    for _ in range(int(steps)):
        i_l4 = rng.randn(col.cfg.n_l4).astype(np.float64) * float(input_scale)
        sp = col.step(i_l4)
        all_spikes_l23.append(sp['L23'])
        all_spikes_l4.append(sp['L4'])
        all_spikes_l5.append(sp['L5'])
        all_spikes_l6.append(sp['L6'])

    s_l23 = np.stack(all_spikes_l23, axis=0)
    s_l4 = np.stack(all_spikes_l4, axis=0)
    s_l5 = np.stack(all_spikes_l5, axis=0)
    s_l6 = np.stack(all_spikes_l6, axis=0)

    aval = detect_avalanches(s_l23)
    aval_alpha = fit_power_law_exponent([float(x) for x in aval], xmin=1.0) if aval else 0.0

    return {
        'spikes': {
            'L23': s_l23,
            'L4': s_l4,
            'L5': s_l5,
            'L6': s_l6,
        },
        'avalanches': aval,
        'avalanche_alpha': float(aval_alpha),
    }
