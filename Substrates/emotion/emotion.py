import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nn import Module, MLP, Linear, AdaptiveNorm, Tensor


# Indices for Panksepp primary systems
PANKSEPP_SYSTEMS = (
    'SEEKING',
    'RAGE',
    'FEAR',
    'LUST',
    'CARE',
    'PANIC',
    'PLAY',
)

BASIC_EMOTIONS = (
    'joy',
    'sadness',
    'anger',
    'fear',
    'disgust',
    'surprise',
)


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _safe_norm01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    s = float(np.sum(v))
    if (not np.isfinite(s)) or s <= 1e-12:
        return np.ones_like(v) / max(1, v.size)
    return v / s


@dataclass
class EmotionState:
    """Full emotion state vector s ∈ R^13 as in the plan.

    - vad: [V,A,D] in [-1,1]
    - panksepp: 7 primary system activations in [0,1]
    - velocity: [dV,dA,dD] (unbounded but kept stable via dynamics)
    - resource: regulation resource R in [0,1]
    """

    vad: np.ndarray
    panksepp: np.ndarray
    velocity: np.ndarray
    resource: float

    def as_vector13(self) -> np.ndarray:
        vad = np.asarray(self.vad, dtype=float).reshape(3)
        p = np.asarray(self.panksepp, dtype=float).reshape(7)
        vel = np.asarray(self.velocity, dtype=float).reshape(3)
        return np.concatenate([vad, p, vel], axis=0)


class AGIEmotionEngine(Module):
    """AGI-grade emotional dynamical system.

    This is an intentionally *mechanistic* engine:
    - subcortical primary-process dynamics (Panksepp 7)
    - continuous VAD dynamics
    - executive regulation with resource consumption
    - basic emotion classification head

    Multimodal feature extraction is expected to be handled upstream (encoder/observe).
    You provide *inputs* (drives/appraisal) as vectors.
    """

    def __init__(
        self,
        vad_dim: int = 3,
        panksepp_dim: int = 7,
        dt_default: float = 0.1,
        label: str = 'emotion_engine',
    ):
        self.label = str(label)
        self.vad_dim = int(vad_dim)
        self.panksepp_dim = int(panksepp_dim)
        self.dt_default = float(dt_default)

        if self.vad_dim != 3:
            raise ValueError('AGIEmotionEngine expects vad_dim=3')
        if self.panksepp_dim != 7:
            raise ValueError('AGIEmotionEngine expects panksepp_dim=7')

        # Core cross-system interaction matrix W (from plan)
        self.W = np.array([
            [0.2, -0.4, -0.3, 0.1, 0.3, -0.2, 0.5],
            [-0.3, 0.0, 0.4, -0.2, -0.4, 0.2, -0.1],
            [-0.4, 0.3, 0.0, -0.1, -0.3, 0.5, -0.2],
            [0.1, -0.1, -0.2, 0.0, 0.2, -0.1, 0.1],
            [0.3, -0.5, -0.4, 0.2, 0.0, 0.4, 0.3],
            [-0.2, 0.2, 0.4, -0.1, 0.3, 0.0, -0.3],
            [0.4, -0.2, -0.3, 0.1, 0.2, -0.2, 0.0],
        ], dtype=float)

        # Time constants tau_i (plan default; can be tuned)
        self.tau = np.array([1.0, 0.8, 0.5, 2.0, 1.5, 0.6, 1.2], dtype=float)

        # Regulation resource dynamics
        self.R_max = 1.0
        self.lambda_recover = 0.05
        self.R_critical = 0.2

        # Basic-emotion VAD prototypes (means + diagonal covariance) (from plan)
        self.basic_means = np.array([
            [0.76, 0.48, 0.35],
            [-0.63, -0.27, -0.33],
            [-0.51, 0.59, 0.25],
            [-0.64, 0.60, -0.43],
            [-0.60, 0.11, -0.10],
            [0.08, 0.78, -0.09],
        ], dtype=float)
        self.basic_cov_diag = np.array([
            [0.12, 0.15, 0.18],
            [0.10, 0.08, 0.14],
            [0.15, 0.20, 0.12],
            [0.14, 0.22, 0.16],
            [0.11, 0.13, 0.15],
            [0.20, 0.18, 0.22],
        ], dtype=float)

        # Map Panksepp to VAD influence (from plan's pseudocode)
        self.M_p2vad = np.array([
            [0.5, -0.6, -0.7, 0.3, 0.4, -0.5, 0.6],
            [0.7, 0.8, 0.9, 0.6, 0.3, 0.4, 0.8],
            [0.4, 0.2, -0.6, 0.1, 0.3, -0.4, 0.2],
        ], dtype=float)

        # Executive regulation controller (trainable) for target tracking
        # Input: [vad, panksepp, velocity, target_vad, resource]
        self.regulator = MLP(
            3 + 7 + 3 + 3 + 1,
            [64, 64, 3],
            label=f'{label}_regulator',
        )

        # Learned appraisal fusion (optional): turns arbitrary appraisal features into (I_ext(7), appraisal_vad(3))
        # If you already compute these upstream, you can bypass by passing them directly.
        self.appraisal_fuser = MLP(
            16,
            [64, 64, 10],
            label=f'{label}_appraisal_fuser',
        )

        # Competitive normalization for Panksepp dynamics (replaces softmax)
        self.panksepp_competition = AdaptiveNorm(7, label=f'{label}_panksepp_comp')

        # Classification distribution over basic emotions
        self.basic_norm = AdaptiveNorm(6, label=f'{label}_basic_norm')

        self.reset()

    def reset(self) -> None:
        self.state = EmotionState(
            vad=np.zeros(3, dtype=float),
            panksepp=np.ones(7, dtype=float) * (1.0 / 7.0),
            velocity=np.zeros(3, dtype=float),
            resource=1.0,
        )

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # Stable sigmoid
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _panksepp_step(self, p: np.ndarray, I_ext: np.ndarray, dt: float) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(7)
        I_ext = np.asarray(I_ext, dtype=float).reshape(7)

        # Primary activation + competitive inhibition
        x = (self.W @ p) + I_ext - 0.5
        act = self._sigmoid(x)

        # Pairwise competition term (kept simple but stable)
        # inhibition_i = sum_{k!=i} alpha_{ik} p_k p_i ; use alpha from |W| scaled
        alpha = np.abs(self.W)
        inhibition = (alpha @ p) * p

        dp = (-p + act - inhibition) / (self.tau + 1e-9)
        p_new = p + dt * dp
        p_new = _clip(p_new, 0.0, 1.0)

        # Competition normalization via AdaptiveNorm (AGI-grade softmax replacement)
        try:
            p_new = self.panksepp_competition(Tensor(p_new)).data
            p_new = np.asarray(p_new, dtype=float).reshape(7)
            p_new = _clip(p_new, 0.0, 1.0)
        except Exception:
            pass

        return p_new

    def _vad_step(self, vad: np.ndarray, p: np.ndarray, appraisal_vad: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        vad = np.asarray(vad, dtype=float).reshape(3)
        p = np.asarray(p, dtype=float).reshape(7)
        appraisal_vad = np.asarray(appraisal_vad, dtype=float).reshape(3)

        dvad = (self.M_p2vad @ p) + appraisal_vad - 0.1 * vad
        vad_new = _clip(vad + dt * dvad, -1.0, 1.0)
        return vad_new, dvad

    def _regulate(self, state: EmotionState, target_vad: np.ndarray, dt: float) -> Tuple[EmotionState, np.ndarray]:
        target_vad = np.asarray(target_vad, dtype=float).reshape(3)

        x = np.concatenate([
            np.asarray(state.vad, dtype=float).reshape(3),
            np.asarray(state.panksepp, dtype=float).reshape(7),
            np.asarray(state.velocity, dtype=float).reshape(3),
            target_vad,
            np.array([float(state.resource)], dtype=float),
        ], axis=0)

        u = self.regulator(Tensor(x)).data
        u = np.asarray(u, dtype=float).reshape(3)

        # Cost and recovery
        regulation_cost = float(np.sum(np.abs(u))) * 0.1
        resource = float(state.resource)
        resource = max(0.0, resource - regulation_cost + self.lambda_recover * (self.R_max - resource) * dt)
        resource = float(_clip(np.array([resource]), 0.0, 1.0)[0])

        # Deep vs surface acting
        gain = (resource / self.R_max) if resource > self.R_critical else 0.1
        vad_reg = _clip(np.asarray(state.vad, dtype=float) + (gain * u * dt), -1.0, 1.0)

        new_state = EmotionState(
            vad=vad_reg,
            panksepp=state.panksepp,
            velocity=state.velocity,
            resource=resource,
        )
        return new_state, u

    def classify_basic_emotion(self, vad: Optional[np.ndarray] = None) -> Tuple[str, np.ndarray]:
        vad = self.state.vad if vad is None else np.asarray(vad, dtype=float).reshape(3)

        # Gaussian score with diagonal covariance: exp(-0.5 * sum((d^2)/var))
        diff = vad.reshape(1, 3) - self.basic_means
        var = self.basic_cov_diag
        var = np.maximum(var, 1e-6)
        logits = -0.5 * np.sum((diff * diff) / var, axis=1)

        # Convert to distribution using AdaptiveNorm
        try:
            probs = self.basic_norm(Tensor(logits)).data
            probs = np.asarray(probs, dtype=float).reshape(6)
            probs = np.clip(probs, 0.0, 1.0)
            probs = _safe_norm01(probs)
        except Exception:
            # Fallback to argmax one-hot
            k = int(np.argmax(logits))
            probs = np.zeros(6, dtype=float)
            probs[k] = 1.0

        label = BASIC_EMOTIONS[int(np.argmax(probs))]
        return label, probs

    def fuse_appraisal(self, appraisal_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert arbitrary appraisal features -> (I_ext(7), appraisal_vad(3)).

        This is optional; if you already compute drives upstream, skip this.
        """
        a = np.asarray(appraisal_features, dtype=float).reshape(-1)
        if a.size != 16:
            a = np.resize(a, 16)
        out = self.appraisal_fuser(Tensor(a)).data
        out = np.asarray(out, dtype=float).reshape(-1)
        if out.size != 10:
            out = np.resize(out, 10)
        I_ext = out[:7]
        appraisal_vad = out[7:10]

        # Constrain: drives in [0,1], appraisal_vad in [-1,1]
        I_ext = _clip(0.5 + 0.5 * np.tanh(I_ext), 0.0, 1.0)
        appraisal_vad = _clip(np.tanh(appraisal_vad), -1.0, 1.0)
        return I_ext, appraisal_vad

    def step(
        self,
        I_ext: Optional[np.ndarray] = None,
        appraisal_vad: Optional[np.ndarray] = None,
        target_vad: Optional[np.ndarray] = None,
        appraisal_features: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Advance emotion dynamics by one step.

        Inputs:
        - I_ext: (7,) external drive for Panksepp systems
        - appraisal_vad: (3,) appraisal impulse on VAD
        - target_vad: (3,) desired VAD for regulation
        - appraisal_features: (16,) optional feature vector to be fused into I_ext/appraisal_vad
        - dt: timestep

        Returns diagnostics dict.
        """
        dt = self.dt_default if dt is None else float(dt)

        if appraisal_features is not None and (I_ext is None or appraisal_vad is None):
            I_f, a_f = self.fuse_appraisal(appraisal_features)
            if I_ext is None:
                I_ext = I_f
            if appraisal_vad is None:
                appraisal_vad = a_f

        if I_ext is None:
            I_ext = np.zeros(7, dtype=float)
        if appraisal_vad is None:
            appraisal_vad = np.zeros(3, dtype=float)
        if target_vad is None:
            target_vad = np.asarray(self.state.vad, dtype=float)

        s0 = self.state
        p1 = self._panksepp_step(s0.panksepp, I_ext, dt)
        vad1, dvad = self._vad_step(s0.vad, p1, appraisal_vad, dt)

        s1 = EmotionState(
            vad=vad1,
            panksepp=p1,
            velocity=dvad,
            resource=float(s0.resource),
        )
        s2, u_reg = self._regulate(s1, target_vad, dt)

        self.state = EmotionState(
            vad=np.asarray(s2.vad, dtype=float).reshape(3),
            panksepp=np.asarray(s2.panksepp, dtype=float).reshape(7),
            velocity=np.asarray(s2.velocity, dtype=float).reshape(3),
            resource=float(s2.resource),
        )

        basic_label, basic_probs = self.classify_basic_emotion(self.state.vad)

        out = {
            'vad': np.asarray(self.state.vad, dtype=float),
            'panksepp': np.asarray(self.state.panksepp, dtype=float),
            'velocity': np.asarray(self.state.velocity, dtype=float),
            'resource': float(self.state.resource),
            'basic_label': basic_label,
            'basic_probs': np.asarray(basic_probs, dtype=float),
            'I_ext': np.asarray(I_ext, dtype=float).reshape(7),
            'appraisal_vad': np.asarray(appraisal_vad, dtype=float).reshape(3),
            'target_vad': np.asarray(target_vad, dtype=float).reshape(3),
            'u_reg': np.asarray(u_reg, dtype=float).reshape(3),
            'state13': self.state.as_vector13(),
        }

        return out

    def parameters(self):
        params = []
        params.extend(self.regulator.parameters())
        params.extend(self.appraisal_fuser.parameters())
        params.extend(self.panksepp_competition.parameters())
        params.extend(self.basic_norm.parameters())
        return params
