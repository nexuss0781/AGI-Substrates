---
description: Active Inference Complete Plan
---

# Active Inference Complete Plan

This document is a **comprehensive, concrete implementation plan** to refactor the current codebase so that:

1. **Active Inference becomes canonical and trainable** (explicit generative model `A/B/C/D`, variational inference / VFE minimization, EFE evaluation, policy posterior `Q(π)`), using only the repo’s internal modules (`nn.py`, etc.).
2. The **16‑dim action vector is grounded** as a real *cognitive action bus* that changes behavior in `mind/brain.py` every tick.
3. **Reasoning owns cognition** (goals, plans, tool choice, learning strategy). Active Inference becomes the normative controller selecting policies/actions under uncertainty and preferences.

The plan is designed to be **non-toy** and **trainable**, while remaining compatible with the existing architecture:

- `mind/brain.py` orchestrates perception/encoding/memory/reasoning and calls active inference.
- `reasoning.py` provides cognition outputs (goals/preferences/constraints/candidate actions).
- `active_inference_upgrades.py` becomes the canonical Active Inference “engine” (belief update + policy posterior).

---

## 0) Current baseline (what is true right now)

### 0.1 `mind/brain.py`

- `AGIMind.tick()` builds a vector currently named `brain_state` using:
  - reasoning latent (if present) else observation latent
  - time features
  - uncertainty in last dim
- `tick()` calls `ActiveInferenceUpgradesFacade.act(brain_state, goal=None, horizon=5)` and returns the chosen 16‑dim action as `out['action']`.
- `tick()` logs transitions to `observe_transition(...)` and trains via `train_step(...)`.
- **The 16‑dim action is not grounded** into cognitive control (recall depth, tool routing, deliberate vs act, etc.)

### 0.2 `active_inference_upgrades.py`

- `ActiveInferenceUpgradesFacade.act(state, goal=None, horizon=5)`:
  - treats `state` as fully sufficient “state”
  - plans using `engine.planner.plan_with_mcts(s, g, horizon)`
  - returns the first action
- There is an EFE calculator (`AGIGradeEFECalculator`), but it is not used as a canonical `Q(π) ∝ softmax(-G(π))` selection rule.
- There is no explicit variational belief `Q(s)` updated by minimizing variational free energy.

### 0.3 `reasoning.py`

- Reasoning is a sophisticated multi-capability system.
- The `UltimateAGICognitiveCore.reason(...)` returns reasoning results and capability information.
- **It does not expose a stable preference prior vector** (`C` / preferred outcomes) intended for Active Inference.

---

## 1) Target architecture (canonical Active Inference + grounded action)

### 1.1 Key roles

- **Reasoning (cognition owner)**
  - maintains goals, generates plans, chooses tools, decomposes tasks
  - outputs *preferences/prior outcomes* + constraints + (optionally) candidate actions/policies

- **Active Inference (normative controller)**
  - maintains explicit belief state `Q(s_t)` (variational posterior)
  - maintains explicit preferences `C` over outcomes `o`
  - evaluates candidate policies using Expected Free Energy `G(π)`
  - computes policy posterior `Q(π) ∝ softmax(-G(π))`
  - returns action `a_t` (16‑dim) as the selected action to execute

- **Brain (orchestrator)**
  - builds evidence `o_t` from perceptual/latent features
  - calls reasoning to produce preferences/constraints/candidates
  - calls active inference to update beliefs + select action
  - *grounds the 16‑dim action into cognitive controls* (recall depth, deliberation, tool routing, etc.)

### 1.2 Canonical POMDP objects (explicit)

We implement an explicit POMDP-like generative model using your existing neural primitives:

- **Hidden state**: `s_t ∈ R^{state_dim}`
- **Observation / evidence**: `o_t ∈ R^{obs_dim}` (in this codebase, evidence is built from encoding/reasoning latents; `obs_dim` can equal `state_dim` to start)
- **Action**: `a_t ∈ R^{16}`

Model parameters:

- **A** likelihood: `p(o_t | s_t)`
- **B** transitions: `p(s_{t+1} | s_t, a_t)`
- **C** preferences: prior over outcomes `p_C(o_t)`
- **D** prior over initial states `p(s_0)`

All will be trainable with your `nn.Tensor` framework.

---

## 2) Concrete code changes by file

### 2.1 `active_inference_upgrades.py` — implement canonical Active Inference engine

#### 2.1.1 Add a new canonical engine class (recommended: keep facade)

Add a new class inside `active_inference_upgrades.py`:

- `class CanonicalActiveInferenceEngine(Module):`

This engine will be **the authoritative canonical implementation**.

It will contain:

- **Generative model** (trainable)
  - `A_net`: `MLP(state_dim, [...], obs_dim * 2)` output `(μ_o, logσ_o^2)`
  - `B_net`: `MLP(state_dim + action_dim, [...], state_dim * 2)` output `(μ_s_next, logσ_s_next^2)`
  - `D_prior`: `(μ_D, logσ_D^2)` vectors (trainable tensors or fixed zeros)

- **Variational posterior** (trainable via inference steps)
  - `qs_mu`: current posterior mean
  - `qs_logvar`: current posterior log variance

- **Preferences**
  - `C_mu`: preferred observation mean (vector)
  - `C_logvar` or preference precision scalar (configurable)

- **Policy evaluation**
  - `policy_proposer`: reuse your planner’s policy network + MCTS rollouts
  - `efe_computer`: a canonical EFE computation (risk + ambiguity + epistemic)

#### 2.1.2 Implement variational inference `infer_state(o_t, a_{t-1})`

Add method:

- `infer_state(self, observation: np.ndarray, prev_action: Optional[np.ndarray]) -> Dict[str, Any]`

Algorithm (trainable, canonical):

1. Build prior `p(s_t)`:
   - If no previous belief: use `D_prior`.
   - Else use `B_net(qs_mu_prev, prev_action)` to produce prior `(μ_prior, logvar_prior)`.

2. Minimize variational free energy by gradient descent on posterior params:

- Define a local posterior tensor variables `μ_q, logvar_q` initialized from prior.
- For `K` inference steps:
  - Sample `s ~ q(s)` (reparameterization using `μ_q, logvar_q`)
  - Predict `o_hat` via `A_net(s)` giving `(μ_o, logvar_o)`
  - Compute negative log-likelihood term: `-log p(o|s)` under Gaussian
  - Compute KL term: `KL(q(s) || p(s))` (Gaussian KL closed form)
  - `F = NLL + KL`
  - Backprop through `Tensor` and update `μ_q, logvar_q` with an inference learning rate

3. Store `qs_mu, qs_logvar` as the new posterior.

Return dict:

- `qs_mu`, `qs_var`, `free_energy`, `prior_mu`, `prior_var`, diagnostics.

This is not a toy: it is the standard variational update used in practical neural active inference.

#### 2.1.3 Implement canonical preferences interface

Add methods:

- `set_preferences(self, C_mu: np.ndarray, C_logvar: Optional[np.ndarray] = None)`
- `update_preferences_from_reasoning(self, pref_vec: np.ndarray)`

The preference prior will be used in EFE risk term.

#### 2.1.4 Implement policy evaluation `compute_G(policy)`

Represent a policy `π` as a sequence of actions `[a_t, a_{t+1}, ..., a_{t+H-1}]`.

Rollout under the generative model:

- Start from posterior `q(s_t)`.
- For each step τ:
  - Predict next state via `B_net` distribution
  - Predict observation via `A_net` distribution

Compute:

- **Risk**: divergence between predicted observation distribution and preference prior `C`.
  - Use Gaussian KL: `KL(q(o_τ) || p_C(o_τ))`
- **Ambiguity**: expected entropy of likelihood `H[p(o|s)]` (from `A_net` variance).
- **Epistemic value** (information gain): approximate via reduction in state uncertainty expected after observation.
  - Practical approximation: `IG ≈ KL(q(s_τ) || p(s_τ))` improvement; or difference between prior and posterior entropies.

Then:

- `G(π) = Στ (risk + ambiguity - epistemic)` with discount.

#### 2.1.5 Implement policy posterior

Compute:

- `q_pi = softmax(-G / temperature)`

Select:

- MAP by default, optionally sampling.

Return the first action of the selected policy.

#### 2.1.6 Trainable learning updates

Add method:

- `learn_from_transition(o_t, a_t, o_{t+1})`

Update:

- `A_net` to better predict `o` given inferred `s`.
- `B_net` to better predict next `s` from current `s` and action.

Losses:

- `L_A = NLL(o_t | A(s_t))`
- `L_B = NLL(s_{t+1} | B(s_t, a_t))` where `s_{t+1}` is inferred from `o_{t+1}`.

This is canonical: generative model learns to support inference/control.

#### 2.1.7 Modify `ActiveInferenceUpgradesFacade`

Keep the existing facade for compatibility but refactor it to delegate:

- `self.canonical = CanonicalActiveInferenceEngine(...)`

Change:

- `act(state, goal, horizon)` now treats input as **observation/evidence**, not hidden state.
  - `belief = canonical.infer_state(observation=state, prev_action=self._last_action)`
  - `canonical.set_preferences(goal)` (goal is now C)
  - propose policies using planner + samples
  - compute `G`, compute `q_pi`, select action

- `observe_transition(...)` becomes:
  - `canonical.learn_from_transition(o_t, a_t, o_{t+1})`
  - can keep existing TD/dynamics learner as auxiliary (optional), but canonical model is authoritative.

- `train_step(...)` can remain as a general hook but should call canonical training and return diagnostics.

---

### 2.2 `reasoning.py` — make preferences explicit (reasoning owns cognition)

We will modify the *returned dict* from reasoning so Active Inference gets **real C**.

#### 2.2.1 Standardize a preference output

In `UltimateAGICognitiveCore.reason(...)` (or immediately after routing), ensure the returned result includes:

- `preferences`: `np.ndarray` length `state_dim` (or `obs_dim`)
- `constraints`: dict (optional)
- `candidate_policies`: optional list of action sequences OR candidate actions

How to generate `preferences` without external LLMs:

- Use the goal engine if present:
  - encode current goal description using semantic encoder → vector
- Else:
  - derive preferences from `query_embedding` + `context_embedding` through a small MLP projector inside reasoning core.

This makes preferences trainable and consistent.

#### 2.2.2 Keep reasoning in control of cognition

Reasoning continues to decide:

- goal content
- plan/tool selections
- capability routing

But it now exports:

- what outcomes it prefers (C)
- constraints (safety/hard rules)

---

### 2.3 `mind/brain.py` — ground the 16‑dim action bus and route preferences

#### 2.3.1 Rename semantics: `brain_state` becomes evidence `o_t`

Internally treat the output of `build_brain_state(...)` as:

- `evidence_t` (observation vector), not hidden state.

#### 2.3.2 Ground the 16‑dim action into cognitive controls

Add a method in `AGIMind`:

- `_decode_cognitive_action(self, action: np.ndarray) -> Dict[str, Any]`

Mapping (fixed initial schema, trainable downstream):

- **dims 0–3** Thinking control
  - `think_depth` (0..1 → int range e.g. 1..8)
  - `deliberation_bias` (act-now vs think-more)
  - `capability_auto_select_strength`
  - `reasoning_temperature`

- **dims 4–6** Memory control
  - `recall_k` (0..1 → int 0..10)
  - `remember_strength` (0..1)
  - `consolidation_strength` (0..1)

- **dims 7–9** Tool routing
  - `use_predictive_tool` probability/strength
  - `use_causal_reasoning` strength
  - `use_planning_simulation` strength

- **dims 10–11** Safety/meta
  - `risk_aversion`
  - `uncertainty_sensitivity`

- **dims 12–15** Reserved for future external actuation

All values will be squashed with `sigmoid/tanh` and then discretized where needed.

#### 2.3.3 Apply control in `tick()`

After Active Inference returns `chosen_action`, do:

- `control = _decode_cognitive_action(chosen_action)`
- Modify cognition routing:

1. **Memory recall depth**
   - When calling `reason_text(...)`, set `k_recall = control['recall_k']`.

2. **Remember gating**
   - Gate `remember` based on `remember_strength`.

3. **Reasoning capability selection**
   - Use `allow_auto_select` based on control.

4. **Predictive tool as active sampling**
   - If `use_predictive_tool` is high, call `predict_step(...)` before/after reasoning to gather internal “observations”.

This step is the *actual grounding*: the 16‑dim vector changes the AGI’s cognitive actions.

#### 2.3.4 Wire reasoning preferences into Active Inference

In `tick()` after `reasoning_out`:

- Extract `preferences` if present; else create fallback using reasoning latent.
- Pass to Active Inference as `goal=preferences`.

This makes `goal` in `ActiveInferenceUpgradesFacade.act(..., goal=...)` become canonical `C`.

---

## 3) Policy proposal strategy (non-toy, trainable)

We will use **two sources** of candidate policies:

1. **Planner proposals (existing)**
   - Use `LearnedDynamicsPlanner.plan_with_mcts(...)` to produce a baseline action sequence.

2. **Stochastic policy sampling**
   - Sample `K` action sequences from a policy network (existing or new) to ensure diversity.

Then Active Inference evaluates them via canonical `G(π)` and computes `Q(π)`.

This is practical: you get canonical selection without requiring full brute-force enumeration.

---

## 4) Trainability and learning loop

### 4.1 What is trained?

- `A_net`: observation model
- `B_net`: transition model
- optionally preference projector in reasoning

### 4.2 What gradients are used?

- Inference step gradients update posterior parameters (not model params).
- Learning step gradients update `A_net/B_net` using NLL and optionally free energy.

### 4.3 Where training happens

- In `ActiveInferenceUpgradesFacade.observe_transition(...)`:
  - call canonical `learn_from_transition(o_t, a_t, o_{t+1})`.

- In `train_step(...)`:
  - run batch training updates from stored transition buffer.

This yields a real trainable system.

---

## 5) Expected new outputs (what you will see)

### 5.1 From `brain.tick()` output dict

Add keys (diagnostics, not comments):

- `out['active_inference'] = {...}`
  - `belief_mu`, `belief_var`, `free_energy`
  - `G_values` per policy
  - `q_pi` posterior
  - `selected_policy_index`

- `out['control'] = {...}`
  - decoded control signals

- `out['preferences'] = preferences_vector`

### 5.2 Behavioral effects

- High uncertainty → action selects policies that increase epistemic value → more prediction/reasoning/recall.
- Strong preferences → risk term dominates → action selects policies predicted to match preferred outcomes.
- Safety dims reduce risky tool routes.

---

## 6) Implementation sequence (safe staged rollout)

This is the recommended order to implement without breaking tests:

### Phase 1 — Canonical engine scaffold
1. Add `CanonicalActiveInferenceEngine` with `A/B/C/D` networks.
2. Add `infer_state(...)` with VFE minimization and Gaussian KL/NLL.

### Phase 2 — Canonical policy posterior
3. Add policy proposal and rollout with A/B.
4. Add `compute_G(...)` and `Q(π)` selection.
5. Update facade `.act(...)` to use canonical inference + policy posterior.

### Phase 3 — Learning
6. Add transition buffer and `learn_from_transition(...)` updates for A/B.
7. Integrate into `observe_transition` and `train_step`.

### Phase 4 — Ground the 16‑dim cognitive action
8. Implement `_decode_cognitive_action` in `brain.py`.
9. Apply control in `tick()` to:
   - recall depth
   - remember gating
   - predictive tool routing
   - allow_auto_select for reasoning

### Phase 5 — Reasoning preferences output
10. Update `reasoning.py` to emit `preferences` consistently.

### Phase 6 — Validation
11. Update/add tests in `tests/test_active_inference_upgrades.py` to validate:
   - belief update reduces free energy on repeated same observation
   - `q_pi` is normalized and sensitive to preferences
   - action influences `tick()` control outputs

---

## 7) Files that will be changed (explicit list)

- `active_inference_upgrades.py`
  - Add canonical engine class
  - Refactor `ActiveInferenceUpgradesFacade.act/observe_transition/train_step`

- `mind/brain.py`
  - Grounding decoder
  - Apply cognitive control in `tick()`
  - Pass preferences from reasoning into active inference

- `reasoning.py`
  - Ensure `reason(...)` outputs a `preferences` vector (and optional constraints/candidates)

- `tests/test_active_inference_upgrades.py`
  - Extend to cover canonical belief update + policy posterior + grounding effects

---

## 8) Success criteria (how you will know it is “real”)

**Canonical checks** (must be true):

- `infer_state()` explicitly minimizes `F = NLL + KL` and returns `free_energy`.
- `act()` explicitly computes `G(π)` for multiple policies and forms `Q(π) ∝ softmax(-G)`.
- Preferences are explicit (`C`) and change policy selection.

**Grounding checks** (must be true):

- The 16‑dim action changes:
  - how much memory is recalled
  - whether memory is written
  - whether predictive tool is invoked
  - whether reasoning auto-selects capability

**Trainability checks**:

- `A/B` learn over time from transitions (loss decreases, predictions improve).
- Free energy and/or prediction error trends downward on stable input streams.

---

## 9) Final note on “no approximation”

In practice, *all implemented canonical active inference in neural systems* uses approximations (e.g., amortized inference, sampling). The plan above is **canonical in structure** and **trainable**, using:

- explicit `A/B/C/D`
- explicit variational posterior `Q(s)`
- explicit VFE minimization
- explicit EFE and `Q(π)`

This is the most practical “real canonical implementation” you can run in your codebase without external dependencies.
