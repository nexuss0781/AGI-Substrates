"""
AGI Neuro-Symbolic Grounding
==============================
Trainable bridge between neural latent representations and symbolic terms.
Extracted from reasoning.py — this is an integration concern, not a reasoning
primitive. Provides:
- Latent → Symbol mapping (via learned projection + cosine similarity)
- Symbol → Latent abstraction (nearest neighbor in embedding space)
- Contrastive grounding loss (margin-based)
- Compositional generalization (rule-based program execution)

Also contains UncertaintyQuantifier (Monte Carlo dropout approximation).
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from nn import Tensor, Module, Linear, MLP
from symbolic_primitives import Term


class GroundingMechanism(Module):
    """Trainable bridge between neural latents and symbolic terms."""

    def __init__(self, latent_dim: int, symbol_embedding_dim: int):
        self.latent_dim = latent_dim
        self.symbol_embedding_dim = symbol_embedding_dim
        self.latent_to_symbol = Linear(latent_dim, symbol_embedding_dim, label='l2s')
        self.symbol_embeddings: Dict[Term, Tensor] = {}
        self.temperature = 0.07

    def _resize_latent(self, latent: Tensor) -> Tensor:
        if len(latent.data) != self.latent_dim:
            if len(latent.data) > self.latent_dim:
                latent_data = latent.data[:self.latent_dim]
            else:
                latent_data = np.pad(latent.data, (0, self.latent_dim - len(latent.data)))
            latent = Tensor(latent_data)
        return latent

    def _l2_normalize(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64).reshape(-1)
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float64, copy=False)

    def get_embedding(self, term: Term) -> Tensor:
        if term not in self.symbol_embeddings:
            seed = int(hash((term.name, term.args)) & 0xFFFFFFFF)
            rs = np.random.RandomState(seed)
            vec = rs.randn(self.symbol_embedding_dim).astype(np.float64) * 0.1
            vec = self._l2_normalize(vec)
            self.symbol_embeddings[term] = Tensor(vec, label=f'emb_{str(term)}')
        return self.symbol_embeddings[term]

    def ground_symbol(self, term: Term, latent: Tensor) -> float:
        """Compute grounding score (similarity between projected latent and symbol embedding)."""
        latent = self._resize_latent(latent)
        projected = self.latent_to_symbol(latent)
        symbol_emb = self.get_embedding(term)

        p = self._l2_normalize(projected.data)
        s = self._l2_normalize(symbol_emb.data)
        sim = float(np.dot(p, s))

        tau = float(getattr(self, 'temperature', 0.07))
        tau = max(tau, 1e-6)
        return float(sim / tau)

    def abstract_latent(self, latent: Tensor, known_symbols: List[Term]) -> Optional[Term]:
        """Find the closest symbolic term to a neural latent."""
        if not known_symbols:
            return None
        scores = {s: self.ground_symbol(s, latent) for s in known_symbols}
        return max(scores, key=scores.get)

    def compute_loss(self, latent: Tensor, positive_term: Term, negative_terms: List[Term]) -> Tensor:
        """Contrastive margin loss: push positive closer, negatives further."""
        latent = self._resize_latent(latent)
        projected = self.latent_to_symbol(latent)
        pos_emb = self.get_embedding(positive_term)

        p = self._l2_normalize(projected.data)
        pos = self._l2_normalize(pos_emb.data)
        pos_sim = float(np.dot(p, pos))

        margin = 0.2
        loss_val = 0.0
        for neg in negative_terms:
            neg_emb = self.get_embedding(neg)
            neg_sim = float(np.dot(p, self._l2_normalize(neg_emb.data)))
            loss_val += max(0.0, margin + neg_sim - pos_sim)
        return Tensor(np.array(loss_val, dtype=np.float64))

    def parameters(self):
        return self.latent_to_symbol.parameters() + list(self.symbol_embeddings.values())


class CompositionalGeneralizer:
    """Rule-based composition of primitive operations for zero-shot generalization."""

    def __init__(self):
        self.primitives = {
            'add': lambda x, y: x + y,
            'mul': lambda x, y: x * y,
            'relu': lambda x: x.relu(),
            'multiply': lambda x, y: x * y,
            'matmul': lambda x, y: x.matmul(y)
        }
        self.rules: Dict[str, Callable] = {}

    def add_primitive(self, name, op):
        self.primitives[name] = op

    def add_rule(self, name: str, rule_fn: Callable):
        self.rules[name] = rule_fn

    def execute_rule(self, rule_name: str, inputs: Dict[str, Any]) -> Tensor:
        return self.compose(self.rules[rule_name](inputs), inputs)

    def compose(self, program: List[Tuple[str, List[str]]], inputs: Dict[str, Any]) -> Any:
        """Execute a symbolic program by composing primitives."""
        results = {}
        for i, (op, arg_names) in enumerate(program):
            op_fn = self.primitives.get(op)
            if not op_fn:
                raise ValueError(f"Op {op} not found")
            call_args = [inputs[n] if n in inputs else results[n] for n in arg_names]
            results[f'_res_{i}'] = op_fn(*call_args)
            results[f'_result_{i}'] = results[f'_res_{i}']
        return results[f'_res_{len(program) - 1}']


class UncertaintyQuantifier:
    """Quantifies epistemic and aleatoric uncertainty via MC dropout approximation."""

    def __init__(self, model: Module):
        self.model = model

    def quantify_epistemic_uncertainty(self, input_tensor: Tensor, num_samples: int = 10,
                                       param_noise_std: float = 0.001,
                                       input_noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        params = []
        try:
            params = list(self.model.parameters())
        except Exception:
            params = []

        base = [p.data.copy() for p in params]
        try:
            for _ in range(int(max(1, num_samples))):
                if params and param_noise_std > 0:
                    for p in params:
                        p.data = p.data + np.random.normal(0.0, float(param_noise_std), p.data.shape)

                if input_noise_std > 0:
                    noisy_in = Tensor(input_tensor.data + np.random.normal(0.0, float(input_noise_std), input_tensor.data.shape))
                else:
                    noisy_in = input_tensor

                preds.append(np.asarray(self.model(noisy_in).data, dtype=np.float64).copy())

                if params:
                    for p, b in zip(params, base):
                        p.data = b.copy()
        finally:
            if params:
                for p, b in zip(params, base):
                    p.data = b.copy()

        preds_arr = np.stack(preds, axis=0) if preds else np.zeros((1, 1), dtype=np.float64)
        return np.mean(preds_arr, axis=0), np.std(preds_arr, axis=0)

    def get_confidence_bounds(self, pred, e_std, a_std):
        total_std = np.sqrt(e_std ** 2 + a_std ** 2)
        return pred - 2 * total_std, pred + 2 * total_std


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AGI Neuro-Symbolic Grounding - Self Test")
    print("=" * 60)

    # Test GroundingMechanism
    gm = GroundingMechanism(latent_dim=32, symbol_embedding_dim=16)
    latent = Tensor(np.random.randn(32), label='latent')

    cat = Term('cat')
    dog = Term('dog')
    fish = Term('fish')

    score = gm.ground_symbol(cat, latent)
    print(f"\n[1] Grounding score (cat): {score:.4f}")

    best = gm.abstract_latent(latent, [cat, dog, fish])
    print(f"[2] Best abstraction: {best}")

    loss = gm.compute_loss(latent, cat, [dog, fish])
    print(f"[3] Contrastive loss: {loss.data:.4f}")

    # Test CompositionalGeneralizer
    cg = CompositionalGeneralizer()
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    prog = [('add', ['x', 'y']), ('relu', ['_res_0'])]
    result = cg.compose(prog, {'x': a, 'y': b})
    print(f"[4] Composed program result: {result.data}")

    # Test UncertaintyQuantifier
    model = MLP(4, [8, 2], label='test')
    uq = UncertaintyQuantifier(model)
    test_input = Tensor(np.random.randn(4))
    mean, std = uq.quantify_epistemic_uncertainty(test_input, num_samples=5)
    print(f"[5] Uncertainty: mean shape={mean.shape}, std shape={std.shape}")

    print("\n" + "=" * 60)
    print("AGI Neuro-Symbolic Grounding: All Tests Passed!")
    print("=" * 60)
