---
description: Active Inference + Neural Substrate Integrated Plan
---

# Active Inference + Neural Substrate Integrated Plan

This document upgrades the previous `docs/active_inference_complete_plan.md` by integrating a **human-like neural substrate** (spiking / temporally dynamic / plastic / microcircuit-inspired) as a *first-class* component that Active Inference and Reasoning can both use.

**Non-negotiables from you**

- No legacy transformer softmax as the core allocator.
- If any softmax-like normalization is needed (policy posterior, attention routing, gating), we will use **`nn.AdaptiveNorm`**.
- Not minimal: end-to-end architecture that ties together:
  - neural substrate
  - memory + learning
  - reasoning
  - active inference (canonical)

This is still implementable incrementally without breaking the current system.

---

## 0) Glossary (how terms map to your code)

- **Evidence / observation (`o_t`)**: The vector built in `mind/brain.py` (currently `brain_state`). In this plan we treat it as evidence.
- **Latent state (`s_t`)**: A hidden state inside the generative model maintained by canonical active inference.
- **Cognitive action (`a_t`)**: Your 16-dim action bus. In this plan it is both:
  - the action used by the generative model transition (`B`)
  - a decoded control vector that *directly changes cognition* (memory, reasoning routing)
- **Preferences (`C`)**: Desired outcomes expressed as a prior over outcomes (vector distribution), produced by reasoning.

---

## 1) Why a neural substrate is needed for canonical Active Inference

Canonical Active Inference is not only math (`A/B/C/D`, VFE, EFE). It requires a **generative model** that is:

- temporally grounded (not static)
- uncertainty-aware (distributions, not point estimates)
- capable of online learning (plasticity / continual learning)
- capable of stable internal dynamics (homeostasis / normalization)

Your `docs/NEURAL_SUBSTRATE_PLAN.MD` provides exactly the missing substrate requirements:

- spiking / temporally dynamic neurons (LIF / SRM approximations)
- synaptic dynamics and plasticity (STDP, metaplasticity)
- microcircuit motifs (E/I balance, recurrence)

The plan below turns that research outline into an **engineering design that fits your current repo**.

---

## 2) Target system architecture (modules and responsibilities)

### 2.1 Top-level dataflow (per tick)

1. `brain.py` builds **evidence** `o_t`.
2. `reasoning.py` produces:
   - `preferences` (vector prior over outcomes) = `C`
   - `constraints` (safety, tool restrictions)
   - optional `candidate_policies` (optional)
3. `active_inference_upgrades.py` (canonical engine) does:
   - belief update (VFE minimization) => `Q(s_t)`
   - policy evaluation (EFE) => `G(π)`
   - policy posterior `Q(π)` using **`AdaptiveNorm`**
   - selects `a_t` (16-dim)
4. `brain.py` decodes `a_t` into cognitive controls and applies:
   - memory recall depth, remember/write gating, consolidation
   - reasoning capability routing parameters
   - tool-use gating (predict, plan simulation, etc.)
5. Learning:
   - transitions are fed back to update generative model and substrate plasticity

### 2.2 Neural substrate placement

We introduce a **Neural Substrate** module that is the shared foundation used by:

- the Active Inference generative model (`A` and `B` live here)
- memory addressing and consolidation signals
- reasoning’s internal workspace (optional but planned)

This substrate is *not* a transformer. It is a temporally dynamic recurrent substrate with plastic synapses.

---

## 3) Concrete neural substrate design (implementable in this repo)

This translates `NEURAL_SUBSTRATE_PLAN.MD` into minimal *biologically-faithful primitives* that are still implementable with your autograd.

### 3.1 Neuron model choice (engineering compromise)

We do not implement full Hodgkin–Huxley initially (too heavy). We implement:

- **LIF-like membrane dynamics** with discrete timesteps (SRM-like kernel optional)
- **surrogate gradient** spike function for training
- optional continuous “rate-coded spikes” in early phases for stability

Representation per neuron:

- membrane potential `v`
- refractory trace `r`
- spike output `z ∈ {0,1}` (or `z∈[0,1]` in surrogate mode)

### 3.2 Synapse model

We implement synapses as weights with:

- optional short-term plasticity traces (facilitation/depression) later
- an STDP-like local update rule (phase 2+)
- metaplasticity / consolidation mask (your `PlasticLinear` already has pieces)

### 3.3 Microcircuit motif (E/I + recurrence)

Implement a “column” as:

- excitatory population `E`
- inhibitory population `I`
- recurrent connectivity `E→E`, `E→I`, `I→E`

This enforces the stability properties you want (gain control, homeostasis).

### 3.4 Normalization / competition

Anywhere we need competitive allocation, posterior normalization, sparse selection:

- use **`nn.AdaptiveNorm`**

This includes:

- `Q(π)` computation
- attention/routing in substrate or reasoning modules
- memory key selection

No legacy softmax.

---

## 4) Where this lives in the codebase (files and classes)

### 4.1 New module file (required)

Create a new file:

- `neural_substrate.py`

It will implement:

- `class SpikingColumn(Module)`
- `class PlasticSynapseMatrix(Module)` or reuse `PlasticLinear`
- `class Neuromodulator(Module)` (for DA/ACh-like scalars driving plasticity and exploration)
- `class NeuralSubstrate(Module)`

`NeuralSubstrate` will expose:

- `forward(evidence, context=None) -> (latent, stats)`
- `step_plasticity(modulators, local_traces, reward=None)`
- `parameters()`

### 4.2 Reuse from `nn.py`

We will reuse:

- `Tensor`, `Module`, `Linear`, `MLP`
- **`AdaptiveNorm`** for normalization/posteriors
- `PlasticLinear` as the basis for plastic synapses
- `AttentionModule` *only if needed* (it already uses `AdaptiveNorm`)

### 4.3 Modify Active Inference engine to use the substrate

Instead of `A_net` and `B_net` being plain MLPs, we wrap them around the substrate:

- `A` likelihood: `p(o|s)` parameterized by substrate readout
- `B` transition: `p(s'|s,a)` parameterized by substrate dynamics

The substrate becomes the “physics” of your generative model.

---

## 5) Canonical Active Inference with AdaptiveNorm (no softmax)

### 5.1 Belief inference (VFE minimization)

Keep the canonical plan, but replace any implicit categorical softmax usage with either:

- Gaussian beliefs (`q(s)=N(μ,Σ)`), no softmax needed, OR
- if you introduce discrete state factors later: use `AdaptiveNorm` for normalization.

### 5.2 Policy posterior `Q(π)` using AdaptiveNorm

Compute `G(π)` values for candidate policies.

Then:

- `scores = -G` (higher is better)
- `Q(π) = AdaptiveNorm(scores, context=preferences_or_state)`

This gives:

- trainable temperature/sparsity
- goal-directed modulation
- stable normalization

### 5.3 Action selection

- Select MAP policy (argmax) or sample from `Q(π)`.
- Execute the first action `a_t`.

---

## 6) Grounding the 16-dim cognitive action bus (expanded to support substrate)

We keep your earlier grounding plan but extend it to include **neuromodulation** and **plasticity control**, consistent with biological substrate.

### 6.1 Updated 16-dim schema (final initial mapping)

- **dims 0–3: deliberation + reasoning routing**
  - `think_depth`
  - `deliberation_bias` (act-now vs think-more)
  - `capability_auto_select_strength`
  - `reasoning_temperature`

- **dims 4–6: memory control**
  - `recall_k`
  - `remember_strength`
  - `consolidation_strength`

- **dims 7–9: tool / simulation control**
  - `use_predictive_tool`
  - `use_planning_simulation`
  - `use_causal_reasoning`

- **dims 10–12: neuromodulators (substrate + learning)**
  - `dopamine_like` (reward prediction / learning gate)
  - `acetylcholine_like` (uncertainty/attention gate)
  - `norepinephrine_like` (arousal / exploration gain)

- **dims 13–15: safety + stability**
  - `risk_aversion`
  - `uncertainty_sensitivity`
  - `plasticity_safety_gate` (prevents runaway updates)

### 6.2 Where applied in `brain.tick()`

- Reasoning params: depth/temp/auto-select
- Memory subsystem: recall_k, remember_strength, consolidation_strength
- Active Inference: policy sampling diversity scaled by neuromodulators
- Neural substrate: neuromodulator scalars drive plasticity and gain

This makes the 16-dim vector a real internal action with system-wide effect.

---

## 7) Memory + learning integration (deep integration)

### 7.1 Memory addressing uses substrate features

Use substrate latent as the **memory key**:

- `key = substrate_latent`

Memory retrieval uses competitive selection:

- compute similarity scores to stored keys
- normalize selection distribution with **`AdaptiveNorm`** (not softmax)

### 7.2 Consolidation = Active Inference + neuromodulation

Consolidation trigger depends on:

- prediction error / free energy change
- dopamine-like control (dim 10)
- consolidation strength (dim 6)

This links plasticity to salience and reduces catastrophic forgetting.

### 7.3 Learning rules

We implement a hybrid:

- gradient-based updates for global generative models (A/B readouts)
- local STDP-like updates inside substrate synapses (phase 2+)
- metaplasticity masks to protect consolidated knowledge

---

## 8) Reasoning integration (how cognition uses substrate)

Reasoning remains the owner of cognition, but gains access to:

- `substrate_latent` as a working memory / context embedding
- `control` decoded from action to determine:
  - which reasoning capability to run
  - how much recall to do
  - whether to run predictive simulation

Reasoning emits:

- `preferences` vector `C` used by Active Inference.

This enforces the separation of concerns you requested.

---

## 9) Implementation phases (end-to-end but staged)

### Phase 0 — Wiring plan (no behavior change)
- Add `neural_substrate.py` with `NeuralSubstrate` stub returning an MLP latent.
- Ensure `AdaptiveNorm` is the only normalization used for posterior-like distributions.

### Phase 1 — Canonical Active Inference uses substrate
- Replace `A_net/B_net` with substrate-driven likelihood/transition heads.
- Implement belief inference (VFE) and EFE-based policy posterior with `AdaptiveNorm`.

### Phase 2 — Ground cognitive action + neuromodulation
- Implement `_decode_cognitive_action` including neuromodulator dims.
- Apply controls to reasoning, memory, and substrate.

### Phase 3 — Plasticity
- Add local plasticity traces (STDP-inspired) in substrate synapses.
- Gate plasticity with neuromodulator dims and safety gate.

### Phase 4 — Microcircuit refinement
- Add E/I separation and recurrence stability.
- Add homeostatic scaling.

---

## 10) Explicit file modifications list

- **New**: `neural_substrate.py` (core requirement)
- Modify: `active_inference_upgrades.py`
  - canonical engine now depends on substrate for A/B
  - policy posterior uses `AdaptiveNorm`
- Modify: `mind/brain.py`
  - decode action -> controls + neuromodulators
  - use substrate latent for memory addressing
- Modify: `reasoning.py`
  - output `preferences` consistently
  - consume substrate latent as context

---

## 11) Success criteria (what must be true after implementation)

- **No legacy softmax** in key places:
  - policy posterior
  - routing distributions
  - memory retrieval distribution
  - attention-like components

- **Canonical Active Inference**:
  - explicit `Q(s)` inference reduces free energy on stable observations
  - explicit `Q(π)` computed from `G(π)` via `AdaptiveNorm`

- **Grounded action bus**:
  - 16-dim action demonstrably changes memory/learning/reasoning routing and substrate plasticity

- **Neural substrate matters**:
  - substrate latent drives predictions and memory keys
  - neuromodulation gates plasticity and exploration

---

## 12) Notes about your existing `nn.py` components

- `AdaptiveNorm` already implements:
  - learned temperature
  - differentiable sparsity
  - goal/context modulation
  - stable renormalization

This is exactly what we need for policy posterior normalization and routing.

- Your `PlasticLinear` already contains scaffolding for:
  - consolidation masks
  - metaplasticity rates
  - activation tracking

We will reuse/extend it inside `neural_substrate.py` rather than creating redundant systems.
