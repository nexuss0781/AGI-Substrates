# attention.py — Full Function Documentation

**File:** `attention.py`  
**Total Lines:** 3417  
**Role:** AGI-Grade Attention Substrate — complete cognitive attention system  
**Total Classes:** 24  
**Total Functions/Methods:** 140+  
**Total Standalone Functions:** 5 (tensor helpers)

---

# TENSOR HELPER FUNCTIONS (Lines 26–65)

### `tensor_exp(t: Tensor)` (26–32)
- Computes element-wise `exp` with clip range [−20, 20]
- Backward: `∂L/∂t = exp(t) × ∂L/∂out`

### `tensor_log(t: Tensor)` (34–40)
- Computes element-wise `log` with floor clip at 1e-10
- Backward: `∂L/∂t = (1/t) × ∂L/∂out`

### `tensor_sqrt(t: Tensor)` (42–48)
- Computes element-wise `√t` with floor clip at 0
- Backward: `∂L/∂t = (0.5/√t) × ∂L/∂out`

### `tensor_sigmoid(t: Tensor)` (50–57)
- Computes `σ(t) = 1/(1+exp(−t))` with clip [−20, 20]
- Backward: `∂L/∂t = σ(t)(1−σ(t)) × ∂L/∂out`

### `tensor_abs(t: Tensor)` (59–65)
- Computes element-wise `|t|`
- Backward: `∂L/∂t = sign(t) × ∂L/∂out`

---

# SECTION 1: ADVANCED VARIATIONAL INFERENCE (Lines 67–407)

---

## Class 1: `VariationalPosterior` (71–137)

AGI-grade variational posterior with multi-modal support and amortized inference.

### `__init__(self, dim, num_modes=1)` (76–89)
- Amortized inference MLP encoder: dim → [128, dim×2]
- Mode logits for multi-modal posterior (if num_modes > 1)
- Per-mode learnable means and log-variances

### `encode(self, obs)` (91–96)
- Amortized inference: observation → (mean, log_var) via encoder MLP

### `sample(self, obs=None, mode=0)` (98–109)
- Reparameterization trick: `z = μ + σ × ε` with proper gradient flow
- Supports amortized (from obs) or parametric (from stored stats) sampling

### `kl_divergence(self, prior_mean, prior_log_var, mode=0)` (111–124)
- Full KL[q(s|o) || p(s)] with gradient flow
- `KL = 0.5 × Σ(σ²_q/σ²_p + (μ_q−μ_p)²/σ²_p − 1 + log(σ²_p/σ²_q))`

### `entropy(self, mode=0)` (126–129)
- Differential entropy of Gaussian: `0.5 × log(2πeσ²)`

### `parameters(self)` (131–137)

---

## Class 2: `LearnedPrecisionNetwork` (140–177)

Dynamic precision (inverse uncertainty) that adjusts based on prediction errors.

### `__init__(self, dim)` (145–150)
- Precision MLP: dim×2 → [64, 32, dim]
- Learnable base log-precision Tensor
- Error history buffer (max 100)

### `forward(self, obs, prediction)` (152–169)
- Computes prediction error as context
- Learned precision modulation: base + learned → exp for positive precision
- Stores error history

### `get_uncertainty(self)` (171–174)
- Returns 1/precision (inverse of base precision)

### `parameters(self)` (176–177)

---

## Class 3: `ExpectedFreeEnergyComputer` (180–270)

EFE = Epistemic Value (information gain) + Pragmatic Value (goal achievement).

### `__init__(self, dim, action_dim)` (186–200)
- Transition model MLP: (state+action) → (mean + log_var)
- Observation model MLP: state → (mean + log_var)
- Preference model MLP: state → reward
- Uncertainty estimator MLP

### `compute_efe(self, current_state, action, goal)` (202–238)
- Predicts next state mean and log_var via transition model
- Predicts observation mean and log_var via observation model
- **Epistemic value**: H(prior) − H(posterior) = information gain
- **Pragmatic value**: exp(−||next_state − goal||²)
- **EFE** = −epistemic − pragmatic (minimize to maximize both)

### `_compute_entropy(self, variance)` (240–244)
- `0.5 × (log(variance) + dim × log(2πe))`

### `select_attention_action(self, state, goal, num_actions=8)` (246–263)
- Generates one-hot candidate actions, selects minimum EFE

### `parameters(self)` (265–270)

---

## Class 4: `ActiveInferenceEngine` (273–407)

Hierarchical belief updating with adaptive learning and convergence detection.

### `__init__(self, dim, action_dim, generative_model)` (278–293)
- 3-mode variational posterior, precision network, EFE computer
- 3 hierarchical belief levels with separate posteriors
- Adaptive learning rate controller (base 0.01, min 0.001)

### `compute_vfe(self, obs, prior_mean, prior_log_var, level=0)` (295–312)
- VFE = Accuracy (precision-weighted prediction error) + Complexity (KL divergence)

### `minimize_vfe(self, obs, prior_mean, prior_log_var, steps=10, lr=0.01)` (314–357)
- Iterative VFE minimization with convergence detection (threshold 1e-4)
- Adaptive learning rate: halve if gradient norm > 10, increase if < 0.1

### `hierarchical_belief_update(self, obs, goal, prior_mean, prior_log_var)` (359–385)
- Updates all 3 hierarchical belief levels
- Each level's belief becomes prior for next level (cascading)

### `active_inference_step(self, obs, goal, prior_mean, prior_log_var)` (387–400)
- Full perception (minimize VFE) + action selection (minimize EFE)
- Returns (attention_action, vfe)

### `parameters(self)` (402–407)

---

# SECTION 2: NEURO-SYMBOLIC ATTENTION BRIDGE (Lines 410–743)

---

## Class 5: `LearnedSymbolicRuleExtractor` (414–605)

Hierarchical rule learning with composition, abstraction, and generalization.

### `__init__(self, dim, max_rules=100)` (419–444)
- Rule encoder MLP, rule library (embeddings + symbols)
- 3-level rule hierarchy: primitive → composite → abstract
- Applicability network MLP, composition network MLP, abstraction network MLP

### `extract_rule(self, obs, attention, concept=None, level='primitive')` (446–482)
- Encodes obs+attention to rule embedding
- Creates rule dict: embedding, concept, condition, action, confidence, level, parent_rules
- Evicts least successful rule at same level if library full

### `compose_rules(self, rule1, rule2)` (484–504)
- Combines two rule embeddings via composition_net → composite rule
- Stores sequential action structure

### `abstract_rules(self, rules)` (506–529)
- Abstracts up to 3 rules via abstraction_net → abstract strategy

### `generalize_rule(self, rule, similar_rules)` (531–544)
- Averages embeddings across similar rules, boosts confidence ×1.2

### `_extract_condition(self, obs)` (546–555)
- Top-3 salient feature indices and values

### `_extract_action(self, attention)` (557–564)
- Argmax attention target + strength

### `apply_rules(self, obs)` (566–599)
- Evaluates applicability of each rule (sigmoid > 0.5)
- Weighted vote → normalized attention distribution
- Tracks usage count per rule

### `parameters(self)` (601–605)

---

## Class 6: `ProbabilisticLogicIntegrator` (608–652)

Fuzzy logic integration with neural attention.

### `__init__(self, dim)` (613–615)
### `add_fuzzy_rule(self, condition, consequence, confidence=1.0)` (617–625)
- Adds IF-THEN fuzzy rule with confidence weight

### `evaluate(self, obs)` (627–652)
- Evaluates all fuzzy rules (activation threshold > 0.1)
- Aggregates confidence-weighted consequences

---

## Class 7: `NeuroSymbolicAttentionBridge` (655–743)

Complete bidirectional neuro-symbolic bridge.

### `__init__(self, grounding_mechanism, reasoning_engine, dim)` (659–669)
- Integrates rule extractor + logic integrator
- Neural→Symbolic MLP (dim→32), Symbolic→Neural MLP (32→dim)

### `concept_driven_attention(self, latent, concept)` (671–680)
### `neural_to_symbolic_grounding(self, obs)` (682–693)
### `symbolic_to_neural_grounding(self, concepts)` (695–706)
- Softmax-normalized attention from concept embeddings

### `forward(self, obs, kb)` (708–729)
- **3-source combination**: 40% rule attention + 30% logic attention + 30% symbolic attention
- Normalized output

### `learn_from_experience(self, obs, attention, outcome)` (731–736)
- Extracts new rule if outcome > 0.7

### `parameters(self)` (738–743)

---

# SECTION 3: MULTI-TIMESCALE PREDICTIVE ATTENTION (Lines 746–841)

---

## Class 8: `MultiTimescalePredictiveModule` (750–841)

Predictions at timescales [1, 5, 20] with uncertainty.

### `__init__(self, dim, timescales=[1,5,20])` (755–773)
- Per-timescale predictor MLP: dim×3 → [128, 64, dim×2] (input = state + velocity + acceleration)
- Per-timescale uncertainty MLP

### `forward(self, current_obs)` (775–812)
- Maintains observation history for dynamics computation
- Computes velocity (1st derivative) and acceleration (2nd derivative)
- Per timescale: prediction mean, surprise (|obs−pred|), uncertainty

### `get_integrated_prediction(self, current_obs, weights=None)` (814–833)
- Weighted average across timescales (default: uniform)

### `parameters(self)` (835–841)

---

# SECTION 4: HIERARCHICAL PRECISION (Lines 844–999)

---

## Class 9: `HierarchicalPrecisionController` (848–922)

Multi-level precision with cross-level propagation.

### `__init__(self, dim, levels=3)` (853–873)
- Per-level precision MLP, base log-precisions, cross-level propagation MLPs

### `forward(self, errors, attention)` (875–909)
- Error + attention context → learned precision modulation per level
- Higher attention → higher precision (multiplicative)
- Cross-level propagation (10% blend from lower to higher)

### `get_effective_precision(self, level=0)` (911–913)
### `parameters(self)` (915–922)

---

## Class 10: `LearnedAttractorDynamics` (925–999)

Learned attractor landscape with multi-attractor competition.

### `__init__(self, dim, num_attractors=5, dt=0.1)` (930–950)
- Landscape MLP, learnable attractor centers, adaptive time constant τ MLP
- Exploration noise = 0.01

### `forward(self, salience, temperature=1.0)` (952–988)
- Potential gradient from landscape_net
- Attractor forces: sum of attractive forces toward each center
- Dynamics: `dx/dt = (−∇V + salience − x)/τ × dt + attractor_force × dt + noise`
- Softmax with temperature → attention distribution

### `reset(self)` (990–992)
### `parameters(self)` (994–999)

---

# SECTION 5: OBJECT-BASED ATTENTION (Lines 1002–1280)

---

## Class 11: `ObjectSegmentationModule` (1006–1062)

### `__init__(self, dim, max_objects=10)` (1010–1018)
- Object proposal MLP → centers + scales, feature extractor MLP

### `forward(self, obs)` (1020–1049)
- Generates proposals, filters by scale > 0.1
- Creates Gaussian soft masks around each center
- Extracts features for each object

### `_create_mask(self, center, scale)` (1051–1056)
- Gaussian mask: `exp(−(x−center)²/(2σ²))`

### `parameters(self)` (1058–1062)

---

## Class 12: `ObjectTracker` (1065–1208)

Object tracking with permanence, occlusion handling, and affordance learning.

### `__init__(self, max_objects=10)` (1070–1082)
- Tracked objects dict, occluded objects dict, predicted positions, affordances

### `predict_position(self, obj)` (1084–1091)
- Linear velocity-based prediction

### `update(self, current_objects)` (1093–1198)
- **Feature matching**: 70% feature similarity + 30% spatial proximity (threshold 0.5)
- **Re-identification**: matches occluded objects to new detections (threshold 0.6)
- **Occlusion handling**: maintains predicted position for 10 frames
- **Velocity tracking**: computes per-object velocity from frame-to-frame displacement

### `learn_affordance(self, track_id, affordance)` (1200–1204)
### `get_affordances(self, track_id)` (1206–1208)

---

## Class 13: `ObjectBasedAttention` (1211–1280)

Complete object-based attention system.

### `__init__(self, dim, max_objects=10)` (1215–1223)
- Segmentation module + tracker + salience MLP (features+metadata → 1)

### `forward(self, obs, goal=None)` (1225–1257)
- Segment → track → compute per-object salience (sigmoid) → mask-weighted attention map

### `get_attended_object(self, attention)` (1259–1274)
- Returns object with highest mask-attention overlap

### `parameters(self)` (1276–1280)

---

# SECTION 6: ATTENTION MEMORY & EPISODIC TRACES (Lines 1283–1549)

---

## Class 14: `AttentionMemory` (1287–1473)

Episodic memory with consolidation, schema formation, and hierarchical organization.

### `__init__(self, dim, capacity=1000)` (1292–1311)
- Episode buffer, schema library
- 3-tier hierarchy: episodic, semantic, procedural
- Consolidation every 50 episodes

### `store(self, obs, attention, outcome, context=None)` (1313–1339)
- Stores episode with importance = |outcome|
- Auto-consolidates periodically
- Evicts least important (importance + access_count) when over capacity

### `consolidate_memories(self)` (1341–1353)
- Clusters last 100 episodes, creates schema from clusters ≥ 3 episodes

### `_cluster_episodes(self, episodes)` (1355–1390)
- Pairwise cosine similarity: 60% observation + 40% attention (threshold 0.7)

### `_extract_schema(self, cluster)` (1392–1409)
- Averages observations, attentions, outcomes → prototype schema

### `retrieve_similar(self, obs, k=5, use_schemas=True)` (1411–1442)
- Cosine similarity retrieval across episodes + schemas, updates access count

### `extract_procedural_memory(self, successful_sequences)` (1444–1454)
- Extracts multi-step skills from successful action sequences (≥3 steps)

### `retrieve_successful(self, threshold=0.7, k=10)` (1456–1459)
### `get_statistics(self)` (1461–1473)
- Returns size, mean/std/max/min outcome

---

## Class 15: `WorkingMemoryIntegration` (1476–1549)

Attention-gated working memory with 7 slots (Miller's law).

### `__init__(self, dim, wm_capacity=7)` (1481–1494)
- WM slots (Tensor), age tracker, gating MLP, update MLP

### `forward(self, obs, attention)` (1496–1533)
- Attends observation, computes sigmoid gate per slot
- Gate > 0.5 → update slot (90% old + 10% new)
- Decays old slots (age > 10 → ×0.95)
- WM modulates attention via additive influence

### `get_wm_content(self)` (1535–1537)
### `clear_wm(self)` (1539–1543)
### `parameters(self)` (1545–1549)

---

# SECTION 7: CAUSAL DISCOVERY & INTERVENTION (Lines 1552–1806)

---

## Class 16: `CausalGraphLearner` (1556–1735)

Causal structure learning with PC algorithm and interventional data.

### `__init__(self, dim, max_nodes=20)` (1561–1582)
- Learnable adjacency matrix, node embeddings (32-d), mechanism MLP
- Independence tester MLP, intervention/observation history

### `test_conditional_independence(self, X, Y, Z=None)` (1584–1598)
- Learned independence test via MLP (sigmoid > 0.5 = independent)

### `pc_algorithm(self, obs_history)` (1600–1628)
- Starts with complete graph, removes edges failing independence tests

### `discover_structure(self, obs_history, interventions)` (1630–1668)
- PC algorithm + interventional refinement
- Blends discovered structure with existing (30% new + 70% old)

### `intervene(self, state, node_idx, value)` (1670–1706)
- do(X=value): sets node, propagates through causal parents (threshold 0.3)
- Uses mechanism_net for causal effect computation

### `forward(self, obs)` (1708–1728)
- Computes causal features by aggregating parent effects per node

### `parameters(self)` (1730–1735)

---

## Class 17: `CounterfactualReasoner` (1738–1806)

Pearl's 3-step counterfactual algorithm.

### `__init__(self, causal_graph)` (1742–1743)

### `counterfactual_query(self, obs, intervention, outcome_fn)` (1745–1775)
- Step 1 **Abduction**: infer latent state
- Step 2 **Action**: apply intervention via causal_graph.intervene
- Step 3 **Prediction**: compare factual vs counterfactual outcome
- Returns causal_effect, whether counterfactual is better

### `explain_attention(self, obs, attention, outcome_fn)` (1777–1806)
- Tests 10 counterfactual attention alternatives
- Returns top 5 sorted by causal effect

---

# SECTION 8: PROGRAM SYNTHESIS FOR ATTENTION (Lines 1809–1931)

---

## Class 18: `AttentionProgramSynthesizer` (1813–1931)

Attention program synthesis from demonstrations.

### `__init__(self, dim)` (1818–1840)
- 5 primitive programs: scan, focus, track, compare, search
- Program encoder MLP, decoder MLP, AdaptiveNorm

### `_scan(self, obs)` (1842–1844) — Uniform scanning
### `_focus(self, obs, target_idx=None)` (1846–1852) — One-hot focus on argmax
### `_track(self, obs)` (1854–1859) — Gradient-based tracking
### `_compare(self, obs)` (1861–1868) — Local variance detection
### `_search(self, obs)` (1870–1876) — Outlier/salience detection

### `synthesize_program(self, demonstrations)` (1878–1913)
- Encodes demos → average embedding → decode to top-3 primitives

### `execute_program(self, program_sequence, obs)` (1915–1925)
- Executes sequence of primitives, sums attention, normalizes

### `parameters(self)` (1927–1931)

---

# SECTION 9: THEORY OF MIND (Lines 1934–2128)

---

## Class 19: `TheoryOfMindModule` (1938–2063)

Multi-agent belief modeling with deception detection and recursive reasoning.

### `__init__(self, dim, max_recursion=3)` (1943–1967)
- Belief encoder MLP, attention predictor MLP, recursive_net MLP
- Divergence tracker MLP, deception detector MLP
- Common ground set, per-agent private knowledge

### `infer_belief(self, obs, agent_id=0)` (1969–1972)
### `compute_belief_divergence(self, my_belief, other_belief)` (1974–1978)
### `detect_deception(self, other_obs, other_action, predicted_action)` (1980–1991)
- Detects mismatch between action and predicted action from belief

### `update_common_ground(self, concept, is_shared=True)` (1993–2001)
### `is_common_knowledge(self, concept)` (2003–2005)
### `perspective_taking(self, my_obs, other_obs)` (2007–2021)
- Returns (my_unique, other_unique) observation differences

### `predict_attention(self, belief)` (2023–2027)
### `recursive_reasoning(self, my_obs, other_obs, depth=1)` (2029–2056)
- Depth 1: "What do they attend to?"
- Depth 2: "What do they think I attend to?"
- Depth 3: "What do they think I think they attend to?"

### `parameters(self)` (2058–2063)

---

## Class 20: `MultiAgentAttentionCoordinator` (2066–2128)

Multi-agent attention coordination with communication protocol.

### `__init__(self, dim, num_agents=2)` (2070–2085)
- ToM module, message encoder/decoder MLPs, coordination MLP, AdaptiveNorms

### `coordinate(self, my_obs, other_observations, other_attentions)` (2087–2110)
- Predicts others' attention via ToM (depth=2), combines all observations via coordination_net
- Adds other attentions with 20% weight, normalizes via AdaptiveNorm

### `communicate(self, my_attention)` (2112–2115)
### `receive_message(self, message)` (2117–2120)
### `parameters(self)` (2122–2128)

---

# SECTION 10: CONSCIOUSNESS METRICS & GLOBAL WORKSPACE (Lines 2131–2777)

---

## Class 21: `PredictiveWorldModel` (2139–2199)

Forward/inverse models for imagination and planning.

### `__init__(self, dim, action_dim)` (2143–2157)
- Forward model MLP: (s,a) → s'
- Inverse model MLP: (s,s') → a
- Reward model MLP: s → r
- Uncertainty model MLP

### `imagine_trajectory(self, start_state, actions)` (2159–2168)
- Rollout via forward model

### `plan_with_imagination(self, current_state, goal, horizon=5, num_samples=10)` (2170–2191)
- Random shooting: sample 10 action sequences, select best by goal distance

### `parameters(self)` (2193–2199)

---

## Class 22: `HierarchicalAttentionController` (2202–2275)

Multi-level attention (scene → object → feature) with bidirectional modulation.

### `__init__(self, dim, num_levels=3)` (2206–2229)
- Per-level attention MLP, top-down modulation MLPs, bottom-up salience MLPs, level-specific norms

### `forward(self, obs, goal=None)` (2231–2263)
- Bottom-up pass: obs → progressively abstracted signals
- Top-down pass: goal → progressively detailed modulation
- Integration at each level: bottom-up + top-down → attention via MLP + AdaptiveNorm

### `parameters(self)` (2265–2275)

---

## Class 23: `CuriosityModule` (2278–2344)

Intrinsic motivation through prediction error, novelty, and empowerment.

### `__init__(self, dim)` (2282–2296)
- Predictor MLP, novelty MLP, empowerment MLP, observation history (max 1000)

### `compute_curiosity(self, obs, next_obs)` (2298–2302)
- Prediction error: ||next_obs − predicted_next||²

### `compute_novelty(self, obs)` (2304–2323)
- 1 − max cosine similarity to last 100 observations

### `compute_empowerment(self, obs)` (2325–2327)
### `intrinsic_reward(self, obs, next_obs)` (2329–2337)
- 40% curiosity + 30% novelty + 30% empowerment

### `parameters(self)` (2339–2344)

---

## Class 24: `AttentionChunking` (2347–2425)

Gestalt-principle grouping of attention items.

### `__init__(self, dim, max_chunks=5)` (2351–2359)
- Chunk encoder MLP, similarity MLP

### `chunk_by_similarity(self, obs)` (2361–2398)
- Encodes each position, pairwise similarity > 0.7 → cluster

### `attend_to_chunks(self, obs, base_attention)` (2401–2419)
- Averages attention within each chunk, applies uniformly to all chunk members

### `parameters(self)` (2421–2425)

---

## Class 25: `MetacognitiveMonitor` (2428–2477)

Confidence estimation and error detection for attention decisions.

### `__init__(self, dim)` (2432–2442)
- Confidence MLP (dim×2 → 1), error detector MLP (dim×3 → 1), performance history

### `estimate_confidence(self, obs, attention)` (2444–2448)
### `detect_error(self, obs, attention, outcome)` (2450–2454)
### `should_revise_strategy(self)` (2456–2465)
- True if recent-10 performance < 80% of overall average

### `update_performance(self, outcome)` (2467–2471)
### `parameters(self)` (2473–2477)

---

## Class 26: `GoalDecomposer` (2480–2519)

Decomposes complex goals into ordered subgoals.

### `__init__(self, dim, max_subgoals=5)` (2484–2492)
- Subgoal generator MLP (dim×2 → dim×max_subgoals), ordering MLP

### `decompose(self, current_state, goal)` (2494–2513)
- Generates flat subgoals, reshapes, orders via argsort

### `parameters(self)` (2515–2519)

---

## Class 27: `AttentionPersistence` (2522–2579)

Attention momentum and inhibition of return.

### `__init__(self, dim)` (2526–2541)
- Momentum MLP, inhibition MLP, attention history (max 10)
- Momentum coefficient = 0.7, inhibition coefficient = 0.3

### `forward(self, current_attention, new_attention)` (2543–2573)
- Blends new attention with momentum from previous attention
- Inhibits recently attended locations (last 3 frames)
- ReLU clamp + normalization

### `parameters(self)` (2575–2579)

---

## Class 28: `IntegratedInformationComputer` (2586–2624)

Integrated Information Theory (IIT) — Φ computation.

### `__init__(self, dim)` (2591–2592)
### `compute_phi(self, state, transition_matrix)` (2594–2613)
- Φ = system_entropy − Σ(partition_entropies) (binary partition)
- Clamps Φ ≥ 0

### `_entropy(self, data)` (2615–2620)
- Normalized distribution entropy: `−Σ p log p`

### `is_conscious(self, phi, threshold=0.5)` (2622–2624)

---

## Class 29: `GlobalWorkspace` (2627–2704)

Global Workspace Theory — broadcasts attended content to all cognitive modules.

### `__init__(self, dim, num_modules=10)` (2632–2654)
- Workspace buffer Tensor, per-module encoder/decoder MLPs
- Competition MLP (dim×num_modules → num_modules)

### `compete_for_workspace(self, module_inputs)` (2656–2669)
- Winner-take-all competition among modules

### `update_workspace(self, attended_content, attention)` (2671–2677)
- Exponential moving average: 90% old + 10% new

### `broadcast(self)` (2679–2687)
- Broadcasts workspace to all modules via decoders

### `get_conscious_content(self)` (2689–2691)
### `clear_workspace(self)` (2693–2695)
### `parameters(self)` (2697–2704)

---

## Class 30: `ConsciousnessSubstrate` (2707–2777)

Complete consciousness substrate integrating IIT + GWT.

### `__init__(self, dim, num_modules=10)` (2711–2722)
### `process(self, obs, attention, module_inputs)` (2724–2760)
- Computes Φ, determines consciousness, triggers workspace competition and broadcast if conscious
- Returns phi, is_conscious, winner_module, broadcasts, conscious_content

### `get_consciousness_report(self)` (2762–2777)
- Reports current/mean/max Φ, consciousness ratio, total samples

---

# SECTION 11: META-ATTENTION & STRATEGY LEARNING (Lines 2780–2864)

---

## Class 31: `MetaAttentionController` (2784–2864)

Meta-level strategy selection with 8 attention strategies.

### `__init__(self, dim, num_strategies=8)` (2789–2812)
- Strategy selector MLP + AdaptiveNorm, strategy embeddings
- 8 strategies: bottom_up, top_down, predictive, exploratory, exploitative, balanced, object_based, symbolic
- Per-strategy performance history

### `select_strategy(self, obs, goal)` (2814–2830)
- Context-dependent strategy selection via MLP + AdaptiveNorm

### `update_performance(self, strategy_idx, outcome)` (2832–2838)
### `get_best_strategy(self)` (2840–2846)
### `get_strategy_report(self)` (2848–2858)
### `parameters(self)` (2860–2864)

---

# SECTION 12: MASTER INTEGRATION — `AGIAttentionSubstrate` (Lines 2867–3305)

---

## Class 32: `AGIAttentionSubstrate` (2871–3305)

Complete attention substrate integrating ALL 24+ components.

### `__init__(self, dim, core, config=None)` (2889–2965)
- Instantiates ALL components:
  - Active inference engine, multi-timescale prediction, neuro-symbolic bridge
  - Object-based attention, hierarchical precision, attractor dynamics
  - Causal graph + counterfactual reasoner, program synthesizer
  - Multi-agent coordinator, consciousness substrate, meta-attention
  - Episodic memory, working memory
  - World model, hierarchical attention, curiosity, chunking, metacognition, goal decomposer, persistence
- 8 learnable salience weights (active_inference: 0.25, predictive: 0.15, surprise: 0.15, symbolic: 0.10, object: 0.10, social: 0.10, curiosity: 0.10, hierarchical: 0.05)

### `forward(self, obs, goal, other_agents=None, context=None)` (2967–3189)
**23-step attention processing pipeline:**
1. Goal decomposition into ordered subgoals
2. Meta-attention strategy selection
3. Hierarchical belief updating (3 levels)
4. Active inference (VFE minimization + EFE action selection)
5. Multi-timescale prediction (3 scales)
6. Hierarchical attention (3 levels)
7. Neuro-symbolic reasoning
8. Object-based attention
9. Curiosity-driven intrinsic reward
10. Multi-agent coordination (if other agents present)
11. **Dynamic routing**: context-dependent weight MLP
12. **Competitive integration**: source-vs-source inhibition via competition matrix
13. **Non-linear gating**: sigmoid gate per source
14. **Sparse winner-take-all**: top-10% attention targets enhanced
15. Attractor dynamics
16. Chunking (Gestalt grouping)
17. Persistence (momentum + inhibition of return)
18. Working memory integration
19. Metacognitive confidence estimation
20. Hierarchical precision update
21. Consciousness processing (IIT + GWT)
22. Causal discovery (10% probability per step)
23. Episodic memory storage with auto-consolidation

Returns dict with: attention, vfe, strategy, predictions, surprise, object/social/hierarchical attention, precisions, consciousness metrics, confidence, intrinsic reward, should_revise flag

### `explain_attention(self, obs, attention)` (3191–3216)
- Counterfactual explanations, strategy report, consciousness report, memory stats

### `synthesize_attention_program(self, demonstrations)` (3218–3220)
### `execute_attention_program(self, program, obs)` (3222–3224)

### `integrate_with_core(self)` (3226–3271)
- Monkey-patches core.step to include full attention processing
- Injects conscious content into attended observation

### `parameters(self)` (3272–3305)
- Returns ALL parameters from all 24+ sub-components + dynamic routing + competition matrix + salience weights

---

# SECTION 13: SELF-TEST (Lines 3308–3417)

`__main__` block testing: initialization, forward pass, multi-agent coordination, explanations, program synthesis, memory systems, and core integration.

---

## Summary Table

| # | Section | Class | Lines | Methods | Purpose |
|---|---------|-------|-------|---------|---------|
| 1 | 1 | `VariationalPosterior` | 71–137 | 6 | Multi-modal variational inference |
| 2 | 1 | `LearnedPrecisionNetwork` | 140–177 | 4 | Dynamic precision/uncertainty |
| 3 | 1 | `ExpectedFreeEnergyComputer` | 180–270 | 5 | EFE for action selection |
| 4 | 1 | `ActiveInferenceEngine` | 273–407 | 5 | Hierarchical VFE minimization |
| 5 | 2 | `LearnedSymbolicRuleExtractor` | 414–605 | 9 | Hierarchical rule learning |
| 6 | 2 | `ProbabilisticLogicIntegrator` | 608–652 | 3 | Fuzzy logic integration |
| 7 | 2 | `NeuroSymbolicAttentionBridge` | 655–743 | 7 | Bidirectional neuro-symbolic |
| 8 | 3 | `MultiTimescalePredictiveModule` | 750–841 | 4 | Multi-scale predictions |
| 9 | 4 | `HierarchicalPrecisionController` | 848–922 | 4 | Cross-level precision |
| 10 | 4 | `LearnedAttractorDynamics` | 925–999 | 4 | Attractor competition dynamics |
| 11 | 5 | `ObjectSegmentationModule` | 1006–1062 | 4 | Gaussian object proposals |
| 12 | 5 | `ObjectTracker` | 1065–1208 | 5 | Permanence + occlusion |
| 13 | 5 | `ObjectBasedAttention` | 1211–1280 | 4 | Complete object attention |
| 14 | 6 | `AttentionMemory` | 1287–1473 | 8 | Episodic memory + schemas |
| 15 | 6 | `WorkingMemoryIntegration` | 1476–1549 | 5 | 7-slot working memory |
| 16 | 7 | `CausalGraphLearner` | 1556–1735 | 7 | PC algorithm + interventions |
| 17 | 7 | `CounterfactualReasoner` | 1738–1806 | 3 | Pearl's 3-step counterfactual |
| 18 | 8 | `AttentionProgramSynthesizer` | 1813–1931 | 8 | Program synthesis + 5 primitives |
| 19 | 9 | `TheoryOfMindModule` | 1938–2063 | 9 | Recursive ToM + deception |
| 20 | 9 | `MultiAgentAttentionCoordinator` | 2066–2128 | 5 | Multi-agent coordination |
| 21 | 10 | `PredictiveWorldModel` | 2139–2199 | 4 | Forward/inverse imagination |
| 22 | 10 | `HierarchicalAttentionController` | 2202–2275 | 3 | Scene→object→feature |
| 23 | 10 | `CuriosityModule` | 2278–2344 | 5 | Intrinsic motivation |
| 24 | 10 | `AttentionChunking` | 2347–2425 | 4 | Gestalt grouping |
| 25 | 10 | `MetacognitiveMonitor` | 2428–2477 | 5 | Confidence + error detection |
| 26 | 10 | `GoalDecomposer` | 2480–2519 | 3 | Subgoal generation |
| 27 | 10 | `AttentionPersistence` | 2522–2579 | 3 | Momentum + inhibition of return |
| 28 | 10 | `IntegratedInformationComputer` | 2586–2624 | 4 | IIT Φ computation |
| 29 | 10 | `GlobalWorkspace` | 2627–2704 | 6 | GWT broadcast |
| 30 | 10 | `ConsciousnessSubstrate` | 2707–2777 | 3 | IIT + GWT integration |
| 31 | 11 | `MetaAttentionController` | 2784–2864 | 5 | 8-strategy meta-attention |
| 32 | 12 | `AGIAttentionSubstrate` | 2871–3305 | 6 | **Master integration** (23-step pipeline) |
