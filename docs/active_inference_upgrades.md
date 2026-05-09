# active_inference_upgrades.py — Full Function Documentation

**File:** `active_inference_upgrades.py`  
**Total Lines:** 5618  
**Role:** AGI-Grade Active Inference Engine — critical upgrades across 3 phases (18 components)  
**Total Classes:** 18 (+ 2 data classes)  
**Total Functions/Methods:** 160+  
**Total Standalone Functions:** 3

---

# PHASE 1: CRITICAL UPGRADES (Lines 1–1778)

---

## Class 1: `AGIGradeEFECalculator` (Lines 31–354) — Upgrade 1.1

Proper Expected Free Energy computation with Bayesian model averaging.

### `__init__(self, state_dim, action_dim, num_hypotheses=5)` (43–88)
- Creates `num_hypotheses` hypothesis models, each containing transition MLP, observation MLP, and reward MLP
- Initializes uniform Bayesian hypothesis weights
- Creates goal-conditioned value function MLP (state_dim×2 → 1)
- Creates epistemic value / information gain network
- Creates novelty detector network
- Creates empowerment estimator (mutual information between actions and states)
- Creates social value network for multi-agent scenarios (state_dim×3 → 1)
- Configurable risk aversion parameter (0 = neutral, 1 = averse)
- Creates learned weight controller MLP (not fixed weights) with AdaptiveNorm
- Maintains visited states deque (maxlen=1000) for novelty
- Defines 3 temporal scales: [1, 5, 20] for hierarchical EFE

### `compute_efe(self, trajectory_states, trajectory_uncertainties, actions, goal=None, other_agents=None)` (90–192)
- Computes full EFE: `Σ_t γ^t [w_p×Pragmatic + w_e×Epistemic + w_n×Novelty + w_em×Empowerment + w_s×Social + Risk]`
- Iterates over all hypotheses for Bayesian model averaging
- For each timestep per hypothesis:
  - **Pragmatic**: goal-conditioned value function or homeostasis penalty
  - **Epistemic**: entropy reduction (prior − posterior) = information gained
  - **Novelty**: learned novelty + kernel density estimation
  - **Empowerment**: action-state mutual information via network
  - **Social**: multi-agent social value computation
  - **Risk**: variance penalty weighted by risk_aversion
- Applies temporal discount (γ=0.95)
- Computes Bayesian model average across hypotheses
- Adds L2 action cost penalty
- Returns total EFE and component breakdown dict

### `_compute_entropy(self, state, uncertainty)` (194–199)
- Computes differential entropy of Gaussian: `0.5 × log(2πe × σ²)`
- Ensures positive variance via maximum clipping

### `_compute_posterior_entropy(self, state, uncertainty, h_idx)` (201–218)
- Runs observation model for hypothesis `h_idx` to get predicted mean and log-variance
- Computes precision-weighted posterior combination (prior + observation)
- Returns posterior entropy

### `_compute_novelty_agi(self, state)` (220–244)
- Returns 1.0 if fewer than 10 visited states
- Combines learned novelty network output (60%) with kernel density estimation (40%)
- KDE uses Gaussian kernel with bandwidth=0.1 over visited states buffer
- Caps density-based novelty at 10.0
- Appends current state to visited_states deque

### `_compute_social_value(self, state, other_agents)` (246–262)
- Concatenates self state, other agent state, and mean interaction representation
- Passes through social_value_net

### `compute_hierarchical_efe(self, trajectory_states, trajectory_uncertainties, actions, goal=None)` (264–291)
- Subsamples trajectory at each temporal scale [1, 5, 20]
- Computes EFE at each scale
- Returns dict mapping scale → EFE value

### `update_hypothesis_weights(self, prediction_errors)` (293–307)
- Converts prediction errors to likelihoods via `exp(-error)`
- Performs Bayesian update: `weights = weights × likelihoods / sum`

### `adapt_weights(self, context, performance_feedback)` (309–331)
- Creates feature vector from context + 5 performance metrics
- Passes through weight_controller MLP + AdaptiveNorm
- Stores adaptive weights for next EFE computation

### `parameters(self)` (333–354)
- Aggregates all trainable parameters from hypothesis models, value functions, novelty, empowerment, social, and weight controller networks

---

## Class 2: `TDCreditAssignment` (Lines 361–606) — Upgrade 1.3

Temporal Difference learning with eligibility traces, causal inference, and hindsight replay.

### `__init__(self, state_dim, action_dim, gamma=0.99, lambda_trace=0.9)` (371–397)
- Creates V(s) value network MLP
- Creates Q(s,a) Q-function MLP
- Creates causal attribution network MLP
- Initializes eligibility traces dict, replay buffer (maxlen=10000), hindsight buffer (maxlen=1000), TD error history

### `compute_td_error(self, state, action, reward, next_state, done)` (399–419)
- Computes `δ = r + γV(s') − V(s)` with terminal state handling
- Appends error to history

### `update_value_function(self, state, td_error, learning_rate=0.01)` (421–455)
- Proper gradient-based update via backpropagation through value network
- Stores experience to replay buffer with target and TD error

### `compute_n_step_return(self, trajectory, n=5)` (457–481)
- Computes `G_t^(n) = Σ_{k=0}^{n-1} γ^k × r_{t+k} + γ^n × V(s_{t+n})`
- Returns list of n-step returns for entire trajectory

### `causal_attribution(self, action, state_before, state_after)` (483–500)
- Estimates P(outcome | do(action)) using learned causal network
- Concatenates state_before, action, state_after and passes through causal_net

### `hindsight_experience_replay(self, trajectory, achieved_goal)` (502–518)
- Stores both original experience and relabeled hindsight experience
- Hindsight: "What if the achieved outcome was the goal all along?" — sets reward=1.0

### `update_eligibility_traces(self, state, action, td_error)` (520–541)
- Decays all existing traces by `γλ`
- Prunes traces below 1e-6
- Adds new trace for current state: `e_t = γλe_{t-1} + ∇V(s_t)`

### `counterfactual_credit(self, action_taken, alternative_actions, state, outcome)` (543–573)
- Computes Q-values for taken action and alternatives
- Returns counterfactual advantage = `Q(taken) − mean(Q(alternatives))`

### `batch_update(self, batch_size=32, learning_rate=0.001)` (575–598)
- Samples random batch from replay buffer
- Computes MSE loss between predicted and target values
- Returns average loss

### `parameters(self)` (600–606)
- Returns parameters from value_net, q_net, causal_net

---

## Class 3: `AGISymbolicInterface` (Lines 620–1007) — Upgrade 1.4

Advanced symbolic interface with dynamic vocabulary and compositional semantics.

### `__init__(self, state_dim, initial_vocab_size=1000, max_vocab_size=100000)` (631–672)
- Dynamic vocabulary with word_to_idx, idx_to_word, word_embeddings, word_frequency maps
- Initializes basic vocabulary with 80+ common words
- Creates word encoder MLP, composition network MLP, syntax parser MLP (3 relations), role labeler MLP (7 roles)
- Creates grounding network (symbol → perceptual state)
- Creates pragmatics network (context-dependent meaning)
- Initializes `AGIMultiHeadSelfAttention` for compositional attention (4 heads)
- Initializes `VSABindingSpace` for structured binding
- Context history deque (maxlen=10)

### `_init_basic_vocabulary(self)` (674–698)
- Adds 80+ basic words: special tokens, function words, pronouns, question words, colors, sizes, emotions, verbs, spatial terms, prepositions, numbers, nouns

### `add_word(self, word)` (700–732)
- Returns existing index if word already present (increments frequency)
- If vocabulary full: evicts least frequent word and reuses index
- Initializes new embedding with small random Gaussian (σ=0.01)

### `encode_utterance(self, utterance, context=None)` (734–796)
- Tokenizes to lowercase words
- Dynamically adds unknown words to vocabulary
- Retrieves word embeddings
- **4-step pipeline**:
  1. Syntactic parsing (dependency tree)
  2. Semantic role labeling
  3. Compositional semantics (attention-based, NOT averaging)
  4. Pragmatics (context-dependent interpretation with history)
- Stores result in context history

### `_parse_syntax(self, word_embeddings)` (798–822)
- Pairwise comparison via syntax_parser MLP
- Predicts relation type: none/subject/object
- Returns list of (head_idx, dependent_idx, relation_type)

### `_label_semantic_roles(self, word_embeddings)` (824–850)
- Combines each word embedding with sentence-level context mean
- Predicts one of 7 roles: agent, patient, theme, goal, source, instrument, location

### `_compose_meaning(self, word_embeddings, dependencies, roles)` (852–884)
- Stacks embeddings and applies multi-head self-attention
- Weights agent/patient roles 2× higher
- Returns role-weighted attention combination

### `_apply_pragmatics(self, meaning, context)` (886–905)
- Concatenates meaning, context, and element-wise interaction
- Passes through pragmatics_net for context-dependent interpretation

### `ground_symbol(self, symbol_embedding, perceptual_state)` (907–923)
- Projects symbol to perceptual space via grounding_net
- Computes cosine similarity — returns grounding strength [0,1]

### `decode_to_utterance(self, thought, max_length=20)` (925–991)
- Beam search decoding (beam width=5)
- Scores candidates by dot-product similarity
- Also retrieves top words by similarity to thought
- Filters out special tokens

### `get_vocabulary_size(self)` (993–995)
- Returns current vocabulary size

### `parameters(self)` (997–1007)
- Returns params from word_encoder, composition_net, syntax_parser, role_labeler, grounding_net, pragmatics_net, composition_attention

---

## Class 4: `LearnedDynamicsPlanner` (Lines 1014–1228) — Upgrade 1.5

Ensemble forward models for model-based planning with MCTS.

### `__init__(self, state_dim, action_dim, num_models=5)` (1023–1049)
- Creates ensemble of 5 forward models (each MLP: state+action → mean + log_var)
- Value function MLP, policy network MLP, uncertainty estimator MLP
- MCTS params: c_puct=1.4, num_simulations=50
- Transition buffer deque (maxlen=10000)

### `learn_dynamics(self, state, action, next_state)` (1051–1099)
- Stores transition and trains ensemble with batch size 32
- Per model: forward pass → MSE loss → backprop → gradient descent (lr=0.01)

### `predict_next_state(self, state, action)` (1101–1122)
- Forward pass through all ensemble models
- Returns (ensemble mean, ensemble std as epistemic uncertainty)

### `plan_with_mcts(self, initial_state, goal, horizon=10)` (1124–1169)
- Full MCTS loop: Selection → Expansion → Evaluation → Backpropagation
- Uses UCB for child selection
- Extracts action sequence by following most-visited children

### `_propose_action(self, state)` (1171–1174)
- Generates action from policy network

### `_evaluate_state(self, state, goal)` (1176–1188)
- Combines learned value (70%) with goal distance (30%)

### `rollout_trajectory(self, initial_state, actions)` (1190–1218)
- Rolls out trajectory using learned dynamics
- Computes homeostatic reward + exploration bonus per step

### `parameters(self)` (1220–1228)
- Returns params from all forward models, value function, policy, uncertainty networks

---

## Class 5: `MCTSNode` (Lines 1231–1281) — Dataclass

MCTS tree node with UCB-based child selection.

- `__post_init__`, `is_fully_expanded`, `is_terminal` (depth>50 or value<-100)
- `select_child(c_puct)`: UCB = exploit + c_puct × √(log(parent_visits)/child_visits)
- `add_child`, `update` (increments visit count, accumulates value)

---

## Class 6: `MAMLMetaLearner` (Lines 1296–1564) — Upgrade 1.6

Model-Agnostic Meta-Learning for fast few-shot adaptation.

### `__init__(self, model_dim, task_embedding_dim=64, inner_lr=0.01, outer_lr=0.001)` (1308–1333)
- Task encoder MLP, meta-parameters Tensor, adaptation network MLP
- Meta-gradient accumulator, task statistics dict, per-task experience buffers

### `encode_task(self, support_examples)` (1335–1360)
- Computes mean input and mean target across support set
- Encodes task via task_encoder MLP → task embedding

### `inner_loop_update(self, params, support_examples, num_steps=5)` (1362–1409)
- Performs `num_steps` gradient descent steps on support set
- `adapted_params = params − inner_lr × gradient`

### `outer_loop_update(self, task_batch)` (1411–1464)
- FOMAML: inner loop per task, then compute meta-gradient on query set
- Averages meta-gradient across tasks
- Updates meta-parameters: `θ = θ − outer_lr × meta_grad`

### `fast_adapt(self, task_id, support_examples, num_steps=5)` (1466–1499)
- Encodes task, stores statistics
- Performs inner loop adaptation
- Refines via adaptation_net using task embedding + adapted params

### `meta_train(self, task_distribution, num_iterations=100)` (1501–1518)
- Samples task batches (size 4) and runs outer_loop_update per iteration

### `get_meta_parameters(self)` / `set_meta_parameters(self, params)` (1520–1526)
- Get/set meta-parameter initialization point

### `compute_task_similarity(self, task_id1, task_id2)` (1528–1542)
- Cosine similarity between task embeddings

### `get_task_distribution_stats(self)` (1544–1557)
- Returns num_tasks, embedding mean/std, last meta-gradient norm

### `parameters(self)` (1559–1564)
- Returns meta_params, task_encoder, adaptation_net parameters

---

## Standalone Function: `upgrade_active_inference_engine(engine)` (Lines 1571–1655)

Replaces 6 engine components with Phase 1 upgrades:
1. EFE Calculator → AGIGradeEFECalculator
2. TD Credit Assignment → TDCreditAssignment
3. Symbolic Interface → AGISymbolicInterface
4. Dynamics Planner → LearnedDynamicsPlanner
5. Meta-Learner → MAMLMetaLearner
6. Verifies World Model integration

---

## `__main__` Block (Lines 1669–1778)

Tests all 5 Phase 1 components: EFE Calculator, TD Credit, Symbolic Interface, Dynamics Planner, MAML Meta-Learner.

---

# PHASE 2: HIGH PRIORITY FEATURES (Lines 1781–3996)

---

## Class 7: `PolicyType` (Lines 1786–1791) — Enum

Policy categories: `REACTIVE`, `DELIBERATIVE`, `HABITUAL`, `EXPLORATORY`

## Class 8: `Policy` (Lines 1794–1799) — Data class

Container: `policy_type`, `network`, `embedding`

---

## Class 9: `AttentionBasedPolicyLibrary` (Lines 1805–2102) — Upgrade 2.1

Attention-based policy retrieval, composition, and abstraction.

### `__init__(self, action_dim, state_dim, max_policies=200)` (1817–1852)
- Policy storage by type, multi-head attention for retrieval (4 heads)
- Policy encoder, context encoder, composition network, abstraction network, importance calculator
- Tracks usage, success, and timestamps

### `add_policy(self, policy)` (1854–1866)
- Intelligent eviction if full (least important policy removed)

### `_evict_least_important(self, ptype)` (1868–1890)
- Computes importance for each policy and removes lowest

### `_compute_importance(self, policy)` (1892–1932)
- Learned importance from 10-feature vector: recency, success_rate, usage_freq, execution count, abstraction score, policy length, counterfactual experience count, adaptation count, expected EFE, context availability

### `retrieve_policy(self, state, goal=None, policy_type=None, top_k=5)` (1934–2010)
- Encodes query (state+goal) via context_encoder
- Encodes candidate policies via policy_encoder
- Applies multi-head attention to full stack (query + policies)
- Computes relevance × success_rate × importance scores
- Returns top-k sorted policies

### `compose_policies(self, policies, state)` (2012–2040)
- Combines up to 3 policies via learned composition network (not averaging)

### `abstract_policies(self, policies)` (2042–2077)
- Creates higher-level abstract policy from up to 5 concrete policies
- Sets temporal_abstraction_level = max(inputs) + 1

### `update_policy_success(self, policy_id, success)` (2079–2091)
- Updates rolling success rate from last 20 executions

### `parameters(self)` (2093–2102)

---

## Class 10: `LearnedHierarchicalPlanner` (Lines 2109–2438) — Upgrade 2.2

Neural network level selection, learned subgoal generation, reachability verification.

### `__init__(self, action_dim, state_dim, num_levels=3)` (2122–2160)
- Level selector MLP + AdaptiveNorm, per-level planner MLPs
- Subgoal generator MLP, subgoal library, reachability verifier, constraint checker
- Temporal horizons: [1, 5, 20], experience buffer (maxlen=1000)

### `select_planning_level(self, current, goal, horizon)` (2162–2186)
- Neural level selection (samples from distribution for exploration)

### `plan_hierarchical(self, current_state, goal_state, horizon=20)` (2188–2201)
- Dispatches to primitive or subgoal-based planning by selected level

### `_plan_primitive(self, current, goal, horizon)` (2203–2226)
- Level-0 planner with simple state update loop

### `_plan_with_subgoals(self, current, goal, level, horizon)` (2228–2265)
- Generates learned subgoals, verifies reachability, falls back to alternatives
- Recursively plans to each subgoal at lower level

### `_generate_learned_subgoals(self, start, goal, num_subgoals)` (2267–2297)
- Checks library first, then generates via subgoal_generator with progress ratio
- Verifies constraints on each generated subgoal

### `_verify_reachability(self, start, goal)` (2299–2312)
- Learned reachability network → threshold at 0.5

### `_check_constraints(self, state)` (2314–2322)
- Learned constraint network → threshold at 0.5

### `_generate_alternative_subgoals(self, start, goal, num_subgoals)` (2324–2341)
- Interpolation + noise for diversity when primary subgoals fail

### `_retrieve_from_library(self, start, goal)` (2343–2371)
- Cosine similarity matching against stored start-goal pairs (threshold 0.7)

### `add_to_library(self, start, goal, subgoals, success)` (2373–2390)
- Stores successful sequences, evicts least successful if >100

### `learn_from_experience(self, start, goal, subgoals, success)` (2392–2426)
- Stores experience, adds to library, updates level_selector via policy gradient on success

### `parameters(self)` (2428–2438)

---

## Class 11: `BayesianTheoryOfMind` (Lines 2452–2761) — Phase 2.3

Bayesian belief tracking with particle filters, game theory, and emotion/personality modeling.

### `__init__(self, state_dim=64, num_particles=100)` (2460–2529)
- Particle filter storage (particles + weights per agent)
- Sequential networks: belief_encoder, observation_model, transition_model, confidence_net
- Game-theoretic: payoff_estimator
- Communication planner: communication_value
- Emotion net (8 basic emotions), personality net (Big Five), alignment scorer for coalitions

### `initialize_agent(self, agent_id, prior_state)` (2531–2540)
- Samples particles from Gaussian prior centered on prior_state

### `update_belief(self, agent_id, observation, action=None)` (2542–2599)
- Prediction step (transition model if action available)
- Update step (observation model likelihood × weights)
- Resampling when effective sample size < N/2
- Returns mean belief, variance, confidence, encoding

### `infer_goal(self, agent_id, trajectory)` (2601–2625)
- Inverse optimal control: recency-weighted average direction, extrapolated 10× forward

### `nash_equilibrium(self, my_state, their_state, my_actions, their_actions)` (2627–2684)
- Builds payoff matrix via payoff_estimator
- Iterative best response solver (max 100 iterations, convergence tolerance 1e-4)

### `should_communicate(self, my_belief, their_belief)` (2686–2694)
- Estimates communication value from belief divergence

### `infer_emotion(self, agent_state)` (2696–2710)
- Softmax over 8 emotion logits: joy, sadness, anger, fear, surprise, disgust, trust, anticipation

### `infer_personality(self, agent_states)` (2712–2728)
- Big Five traits from averaged observations: openness, conscientiousness, extraversion, agreeableness, neuroticism (sigmoid normalized to [0,1])

### `find_coalition(self, my_state, agent_states, goal)` (2730–2747)
- Alignment scoring per agent, threshold 0.7

### `parameters(self)` (2749–2761)

---

## Class 12: `GraphMatchingAnalogicalReasoning` (Lines 2769–3132) — Phase 2.4

Graph matching with Hungarian algorithm, structural constraints, schema induction.

### `__init__(self, node_dim=64, edge_dim=32)` (2777–2817)
- GNN node encoder, edge encoder, similarity scorer, schema abstractor, re-representer
- Schema library

### `encode_graph(self, nodes, edges)` (2819–2831)
- Encodes all nodes and edges via respective networks

### `compute_similarity_matrix(self, source_nodes, target_nodes)` (2833–2847)
- Pairwise learned similarity between source and target nodes

### `hungarian_matching(self, similarity)` (2849–2892)
- Kuhn-Munkres algorithm implementation: row/column minima subtraction, zero covering, initial assignment

### `check_structural_constraints(self, matches, source_edges, target_edges)` (2894–2917)
- Verifies if mapping preserves edge structure

### `find_analogy(self, source_nodes, source_edges, target_nodes, target_edges)` (2919–2950)
- Full pipeline: encode → similarity → Hungarian → structural check → confidence

### `induce_schema(self, examples)` (2952–3012)
- Abstracts from examples via schema_abstractor
- K-means clustering (k=min(5, n), 20 iterations) to find prototype

### `apply_schema(self, schema, target_nodes)` (3014–3027)
- Blends encoded nodes with schema prototype (70/30)

### `rerepresent(self, nodes)` (3029–3040)
- Re-representation to enable analogies that require perspective shift

### `cross_modal_analogy(self, visual_features, linguistic_features)` (3042–3057)
- Encodes both modalities to common space, computes similarity

### `progressive_alignment(self, source_nodes, target_nodes, iterations=5)` (3059–3091)
- Greedy iterative matching: easy matches first, then harder ones

### `add_schema(self, schema)` / `retrieve_schema(self, nodes)` (3093–3122)
- Schema library management with cosine similarity retrieval (threshold 0.5)

### `parameters(self)` (3124–3132)

---

## Class 13: `SafeExplorationController` (Lines 3140–3427) — Phase 2.5

Constrained safe exploration with CVaR, Lagrangian optimization, and fallback policies.

### `__init__(self, state_dim=64, action_dim=16)` (3148–3196)
- Safety constraint predictor, risk estimator, uncertainty estimator, CVaR value function
- Information gain estimator, safe fallback policy
- Safety threshold=0.8, risk tolerance=0.1

### `is_safe(self, state, action)` (3198–3212)
- Sigmoid on safety score → threshold at 0.8

### `estimate_risk(self, state, action)` (3214–3226)
- Returns absolute risk score from risk_net

### `estimate_uncertainty(self, state)` (3228–3239)
- Epistemic uncertainty about state

### `compute_cvar(self, state, alpha=0.1)` (3241–3264)
- CVaR using Cornish-Fisher expansion with standard normal quantile
- `CVaR = μ − σ × φ(z_α) / α`

### `information_gain(self, state, action)` (3266–3278)
- Learned information gain from state-action pair

### `select_safe_action(self, state, candidate_actions, values)` (3280–3321)
- Filters candidates by safety, selects highest value among safe ones
- Falls back to safe_policy if none pass

### `risk_sensitive_planning(self, state, candidate_actions, horizon=5)` (3323–3343)
- Maximizes worst-case value: `CVaR − risk_tolerance × risk`

### `exploration_bonus(self, state, action)` (3345–3357)
- Combined 50% uncertainty + 50% information gain

### `constrained_policy_optimization(self, state, candidate_actions, values, constraints)` (3359–3399)
- Augmented Lagrangian with adaptive lambda
- Excludes unsafe actions, applies constraint penalty

### `safe_policy_improvement(self, old_policy_action, new_policy_action, state)` (3401–3416)
- Only accepts improvement if new policy is safe and score ≥ 90% of old

### `parameters(self)` (3418–3427)

---

## Class 14: `StructuralCausalReasoning` (Lines 3435–3913) — Phase 2.6

Pearl's structural causal models with PC algorithm, do-calculus, and counterfactual reasoning.

### `__init__(self, num_variables=10)` (3443–3489)
- Causal graph adjacency matrix
- Per-variable mechanism networks (Sequential)
- Conditional independence tester, intervention effect estimator, counterfactual reasoner, causal effect estimator
- Data buffer (maxlen=1000)

### `add_data(self, observation)` (3491–3496)
- Stores observations for structure learning

### `test_independence(self, x_idx, y_idx, conditioning_set)` (3498–3570)
- Partial correlation via precision matrix (inverse covariance)
- Fallback to residualization (regress X,Y on Z, correlate residuals)
- Fisher's z-test for significance

### `pc_algorithm(self, alpha=0.05)` (3572–3611)
- Starts with complete undirected graph
- Removes edges based on conditional independence (empty set, then size-1)
- Returns adjacency matrix

### `do_intervention(self, state, intervention_var, intervention_value)` (3613–3648)
- Sets intervened variable, propagates through causal graph 3 iterations using mechanisms

### `counterfactual_reasoning(self, actual_state, intervention_var, intervention_value)` (3650–3700)
- **Pearl's 3-step**: (1) Abduction — gradient optimization (50 iterations) to infer exogenous variables, (2) Action — set intervention, (3) Prediction — counterfactual_net

### `estimate_causal_effect(self, cause_var, effect_var, intervention_value, baseline_state)` (3702–3719)
- ATE = `E[Y|do(X=x)] − E[Y|do(X=x₀)]`

### `learn_mechanism(self, variable_idx, data)` (3721–3771)
- Trains per-variable mechanism with backprop (50 epochs, lr=0.01)

### `backdoor_adjustment(self, cause_var, effect_var, confounders, data)` (3773–3825)
- Stratified estimation: `P(Y|do(X)) = Σ_z P(Y|X,Z)P(Z)`
- Weighted average of regression coefficients across strata

### `frontdoor_adjustment(self, cause_var, effect_var, mediator_var, data)` (3827–3849)
- Causal effect estimation through known mediator

### `instrumental_variable(self, instrument_var, cause_var, effect_var, data)` (3851–3902)
- Two-stage least squares (2SLS): Stage 1 regress X on Z, Stage 2 regress Y on X̂

### `parameters(self)` (3904–3913)

---

## Standalone Function: `upgrade_active_inference_engine(engine)` (Lines 3930–3996)

Installs all Phase 1 + Phase 2 components (12 total) into engine.

---

# PHASE 3: ENHANCEMENTS (Lines 4000–5618)

---

## Class 15: `HabitFormationSystem` (Lines 4004–4267) — Phase 3.1

Options framework, skill chunking, and context-dependent habit formation.

### `__init__(self, state_dim=64, action_dim=16)` (4011–4067)
- Option policy, initiation set classifier, termination condition networks
- Context encoder, habit strength tracker, skill library with usage/success tracking
- Chunk detector MLP, hierarchical skill composer
- Decay rate 0.99

### `can_initiate(self, state, skill_id)` / `should_terminate(self, state, skill_id)` (4069–4087)
- Sigmoid thresholded at 0.5

### `execute_skill(self, state, skill_id)` (4089–4100)
- Runs skill policy, increments usage count

### `detect_chunk(self, action_sequence)` (4102–4137)
- Scans consecutive action pairs via chunk_detector
- Extends chunk boundaries greedily

### `create_skill_from_chunk(self, state_sequence, action_sequence)` (4139–4177)
- Creates new Sequential skill network, trains 100 epochs on sequence
- Stores context embedding and initializes habit strength=0.1

### `context_triggered_retrieval(self, state)` (4179–4202)
- Cosine similarity between current context and stored skill contexts
- Weighted by habit strength, sorted by activation

### `update_habit_strength(self, skill_id, success)` (4204–4215)
- Success: +0.1 (capped at 1.0), failure: −0.05 (floor at 0.0)

### `decay_habits(self)` (4217–4220)
- Multiplies all habit strengths by 0.99

### `compose_skills(self, skill_id1, skill_id2)` (4222–4254)
- Creates composed skill from two skill context embeddings via skill_composer

### `parameters(self)` (4256–4267)

---

## Class 16: `EpistemicTheoryOfMind` (Lines 4275–4510) — Phase 3.2

Second-order false beliefs, trust modeling, and pedagogical reasoning.

### `__init__(self, state_dim=64)` (4282–4336)
- First-order belief tracker, second-order belief tracker ("A thinks B thinks...")
- Trust model, teaching strategy network, perspective-taking network
- Belief revision network, surprise detector
- Trust history and belief history per agent

### `infer_first_order_belief(self, agent_state)` (4338–4341)
### `infer_second_order_belief(self, agent_a_state, agent_b_state)` (4343–4352)
### `false_belief_test(self, agent_state, true_state, agent_observation)` (4354–4375)
- Classic Sally-Anne test: compares agent belief with true state (error threshold 0.5)

### `estimate_trust(self, agent_id, agent_state)` (4377–4404)
- Combines historical reliability with neural trust estimate (sigmoid output)

### `update_trust(self, agent_id, prediction, outcome)` (4406–4415)
### `plan_teaching(self, learner_state, target_knowledge)` (4417–4434)
- Bridges knowledge gap between learner's current knowledge and target

### `take_perspective(self, my_state, their_position)` (4436–4445)
### `revise_belief(self, old_belief, new_evidence)` (4447–4456)
### `detect_surprise(self, expected, observed)` (4458–4472)
- Returns (is_surprised, magnitude) with threshold 0.5

### `track_belief_evolution(self, agent_id, belief)` (4474–4483)
### `predict_belief_change(self, agent_id, new_evidence)` (4485–4498)
### `parameters(self)` (4500–4510)

---

## Class 17: `GroundedSymbolicInterface` (Lines 4518–4716) — Phase 3.3

Grounded language with pragmatics, dialogue state tracking, and instruction following.

### `__init__(self, state_dim=64, vocab_size=100000)` (4526–4577)
- Grounding network (word → perceptual state), pragmatics network
- Dialogue state tracker, instruction parser, QA network, compositional generalizer
- Grounding memory dict, dialogue history (max 10)

### `ground_symbol(self, word, perceptual_state)` (4579–4582)
### `retrieve_grounding(self, word)` (4584–4586)
### `interpret_pragmatics(self, utterance, context)` (4588–4597)
### `update_dialogue_state(self, utterance, goal)` (4599–4621)
### `parse_instruction(self, instruction)` (4623–4630)
### `answer_question(self, question, knowledge)` (4632–4641)
### `compositional_generalization(self, concept1, concept2)` (4643–4652)
### `multi_turn_dialogue(self, utterances, goal)` (4654–4674)
### `test_compositional_generalization(self, train_concepts, test_concept)` (4676–4705)
- Measures alignment: positive projection on both component concepts

### `parameters(self)` (4707–4716)

---

## Class 18: `ActiveLearningSystem` (Lines 4724–4950) — Phase 3.4

Information-seeking actions, Bayesian optimization, and experimental design.

### `__init__(self, state_dim=64, action_dim=16)` (4732–4780)
- Info gain estimator, query generator, uncertainty estimator
- Acquisition function network (UCB/EI), experiment designer, optimal design network

### `estimate_information_gain(self, state, action)` (4782–4794)
### `generate_query(self, state)` (4796–4803)
### `estimate_uncertainty(self, state)` (4805–4816)
### `select_informative_action(self, state, candidate_actions)` (4818–4838)
- Returns action maximizing information gain

### `bayesian_optimization_step(self, state, candidate_actions, mean_estimates, variance_estimates)` (4840–4869)
- Learned acquisition function (UCB-style) over mean+variance

### `design_experiment(self, current_knowledge, hypothesis)` (4871–4880)
### `optimal_experimental_design(self, state, num_experiments=5)` (4882–4901)
- Generates sequential experiments with simulated knowledge update

### `add_observation(self, state, action, outcome)` (4903–4912)
### `compute_expected_information_gain(self, state, action, possible_outcomes, outcome_probs)` (4914–4939)
- `EIG = H(Y) − E[H(Y|X)]` via weighted sum over possible outcomes

### `parameters(self)` (4941–4950)

---

## Class 19: `MultiObjectiveOptimizer` (Lines 4958–5181) — Phase 3.5

Pareto frontier computation, learned preference weights, multi-objective policy optimization.

### `__init__(self, state_dim=64, action_dim=16, num_objectives=3)` (4965–5006)
- Per-objective value estimator networks
- Preference learner, context-dependent weight learner, dominance checker
- Pareto archive (max 100)

### `evaluate_objectives(self, state, action)` (5008–5020)
### `is_pareto_dominated(self, values1, values2)` (5022–5032)
### `compute_pareto_frontier(self, candidates)` (5034–5057)
- Brute-force non-dominated sorting

### `scalarize(self, objective_values, weights)` (5059–5069)
### `learn_weights(self, state)` (5071–5083)
- Softmax scalarization weights from state

### `select_from_pareto_frontier(self, pareto_frontier, state)` (5085–5109)
- Learns weights, scalarizes, selects best

### `update_pareto_archive(self, action, objective_values)` (5111–5135)
- Adds if non-dominated, removes dominated archive entries

### `learn_from_preference(self, preferred, rejected)` (5137–5150)
### `multi_objective_policy_optimization(self, state, candidate_actions)` (5152–5171)
- Full pipeline: evaluate → Pareto frontier → select

### `parameters(self)` (5173–5181)

---

## Class 20: `LifelongLearningSystem` (Lines 5189–5502) — Phase 3.6

Continual adaptation with EWC, progressive networks, memory replay, and knowledge distillation.

### `__init__(self, state_dim=64, action_dim=16)` (5197–5238)
- Base network, task-specific adapters list, Fisher information matrix dict
- Progressive columns list with lateral connections
- Replay buffer (max 10000), task embedder
- EWC lambda=1000.0

### `compute_fisher_information(self, data)` (5240–5276)
- Accumulates squared gradients (diagonal Fisher approximation) over data

### `save_optimal_params(self)` (5278–5284)
### `ewc_loss(self)` (5286–5309)
- `0.5 × λ × Σ_task Σ_param F × (θ − θ*)²`

### `add_progressive_column(self)` (5311–5332)
- Adds new column + lateral connections from previous columns

### `forward_progressive(self, state)` (5334–5350)
- Forward through all columns, uses latest output

### `add_to_replay_buffer(self, state, action, task_id)` (5352–5361)
### `sample_replay_batch(self, batch_size=32)` (5363–5371)
### `train_with_replay(self, new_data, learning_rate=0.01)` (5373–5402)
- Mixes new data with 50% replay, trains with EWC penalty

### `create_task_adapter(self, task_id)` (5404–5417)
- Creates small adapter network (modulation signal)

### `forward_with_adapter(self, state, task_id)` (5419–5435)
- Base output + adapter modulation

### `knowledge_distillation(self, student_data, temperature=2.0)` (5437–5470)
- Teacher-student distillation with MSE loss on soft targets

### `switch_task(self, new_task_id, data)` (5472–5489)
- Computes Fisher, saves optimal params, creates adapter for new task

### `parameters(self)` (5491–5502)

---

## Standalone Function: `upgrade_active_inference_engine_complete(engine)` (Lines 5520–5617)

Complete upgrade installing all 18 components across 3 phases:
- **Phase 1 (6):** EFE Calculator, TD Credit, Symbolic Interface, Dynamics Planner, Meta-Learner
- **Phase 2 (6):** Policy Library, Hierarchical Planner, Theory of Mind, Analogical Reasoning, Safe Exploration, Causal Reasoning
- **Phase 3 (6):** Habit Formation, Epistemic ToM, Grounded Language, Active Learning, Multi-Objective Optimizer, Lifelong Learning

---

## Summary Table

| # | Phase | Class | Lines | Methods | Purpose |
|---|-------|-------|-------|---------|---------|
| 1 | 1 | `AGIGradeEFECalculator` | 31–354 | 10 | Bayesian EFE computation |
| 2 | 1 | `TDCreditAssignment` | 361–606 | 10 | TD learning + causal credit |
| 3 | 1 | `AGISymbolicInterface` | 620–1007 | 12 | Dynamic vocab + compositional semantics |
| 4 | 1 | `LearnedDynamicsPlanner` | 1014–1228 | 8 | Ensemble dynamics + MCTS |
| 5 | 1 | `MCTSNode` | 1231–1281 | 6 | MCTS tree node |
| 6 | 1 | `MAMLMetaLearner` | 1296–1564 | 10 | Few-shot meta-learning |
| 7 | 2 | `PolicyType` | 1786–1791 | — | Policy category enum |
| 8 | 2 | `Policy` | 1794–1799 | 1 | Policy data container |
| 9 | 2 | `AttentionBasedPolicyLibrary` | 1805–2102 | 8 | Attention policy retrieval |
| 10 | 2 | `LearnedHierarchicalPlanner` | 2109–2438 | 12 | Learned subgoal planning |
| 11 | 2 | `BayesianTheoryOfMind` | 2452–2761 | 10 | Particle filter ToM |
| 12 | 2 | `GraphMatchingAnalogicalReasoning` | 2769–3132 | 13 | Hungarian algorithm analogies |
| 13 | 2 | `SafeExplorationController` | 3140–3427 | 11 | CVaR + constrained optimization |
| 14 | 2 | `StructuralCausalReasoning` | 3435–3913 | 12 | PC algorithm + do-calculus |
| 15 | 3 | `HabitFormationSystem` | 4004–4267 | 10 | Skill chunking + options |
| 16 | 3 | `EpistemicTheoryOfMind` | 4275–4510 | 13 | Second-order beliefs + trust |
| 17 | 3 | `GroundedSymbolicInterface` | 4518–4716 | 11 | Grounded language + dialogue |
| 18 | 3 | `ActiveLearningSystem` | 4724–4950 | 9 | Information-seeking actions |
| 19 | 3 | `MultiObjectiveOptimizer` | 4958–5181 | 10 | Pareto optimization |
| 20 | 3 | `LifelongLearningSystem` | 5189–5502 | 13 | EWC + progressive nets |
