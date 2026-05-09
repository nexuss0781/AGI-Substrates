# ATTENTION.PY - AGI-GRADE ATTENTION SUBSTRATE DOCUMENTATION

**Total Lines:** 3417  
**Language:** Python  
**Purpose:** Complete AGI-grade attention substrate with active inference, neuro-symbolic reasoning, consciousness metrics, and multi-agent coordination

---

## SECTION 1: FEATURE BREAKDOWN (NON-TECHNICAL)

### Core Attention Mechanisms

**Active Inference Engine**
- The system uses active inference to select where to pay attention based on minimizing surprise and achieving goals
- Implements hierarchical belief updating across 3 levels for multi-scale understanding
- Adaptive learning rates that adjust based on gradient magnitude for stable convergence
- Proper Expected Free Energy computation with epistemic (information gain) and pragmatic (goal achievement) values

**Dynamic Routing System**
- Context-dependent weight computation that adapts attention based on current observation and goal
- Competitive dynamics where strong attention sources can inhibit weaker ones
- Non-linear integration using sigmoid gating for each salience source
- Winner-take-all mechanism creating sparse attention (only top 10% of inputs receive focus)

**Multi-Timescale Prediction**
- Predicts future states at short (1 step), medium (5 steps), and long-term (20 steps) horizons
- Tracks velocity and acceleration for higher-order dynamics
- Uncertainty estimation for each prediction timescale
- Integrated surprise computation across all timescales

### Symbolic and Conceptual Processing

**Neuro-Symbolic Bridge**
- Learns symbolic rules from successful attention patterns
- Hierarchical rule organization: primitive → composite → abstract
- Rule composition to create higher-level strategies
- Bidirectional grounding between neural patterns and symbolic concepts
- Fuzzy logic integration for probabilistic reasoning

**Program Synthesis**
- Learns reusable attention programs from demonstrations
- Library of attention primitives: scan, focus, track, compare, search
- Compositional program construction
- Program execution for consistent attention patterns


### Object and Scene Understanding

**Object-Based Attention**
- Segments observations into distinct objects
- Tracks objects across time with unique IDs
- Object permanence: maintains representation even when occluded
- Predictive tracking during occlusion periods
- Re-identification when objects reappear
- Affordance learning: what can be done with each object

**Hierarchical Attention Controller**
- Multi-level attention processing: scene → object → feature
- Bottom-up salience propagation from low to high levels
- Top-down goal-driven modulation from high to low levels
- Integration at each hierarchical level

### Memory Systems

**Episodic Memory**
- Stores attention episodes with observations, actions, and outcomes
- Automatic memory consolidation into schemas
- Hierarchical organization: episodic, semantic, procedural
- Similarity-based retrieval
- Schema formation from repeated patterns
- Procedural memory extraction from successful sequences

**Working Memory Integration**
- 7-slot working memory capacity (Miller's magic number)
- Gating network decides what to store
- Age-based decay of unused slots
- Working memory modulates attention

### Social and Multi-Agent Features

**Theory of Mind**
- Infers other agents' beliefs from observations
- Belief divergence tracking between self and others
- Deception detection when actions don't match predicted beliefs
- Common ground tracking (shared knowledge)
- Perspective-taking considering sensory differences
- Recursive reasoning up to 3 levels deep

**Multi-Agent Coordination**
- Coordinates attention across multiple agents
- Communication protocol for sharing attention states
- Message encoding and decoding
- Joint attention computation


### Causal Understanding

**Causal Graph Learning**
- PC (Peter-Clark) algorithm for causal structure discovery
- Conditional independence testing
- Interventional data integration
- Learns causal mechanisms between variables
- Maintains intervention history

**Counterfactual Reasoning**
- Three-step algorithm: abduction, action, prediction
- Answers "what if" questions about alternative attention patterns
- Causal effect computation
- Attention explanation through counterfactuals

### Consciousness and Metacognition

**Consciousness Substrate**
- Integrated Information Theory (Φ) computation
- Global Workspace Theory implementation
- Module competition for workspace access
- Broadcasting c

### Causal Understanding

**Causal Graph Learning**
- PC algorithm for causal structure discovery
- Conditional independence testing
- Interventional data integration
- Learns causal mechanisms between variables
- Maintains intervention history

**Counterfactual Reasoning**
- Three-step algorithm: abduction, action, prediction
- Answers what-if questions about alternative attention patterns
- Causal effect computation
- Attention explanation through counterfactuals

### Consciousness and Metacognition

**Consciousness Substrate**
- Integrated Information Theory computation
- Global Workspace Theory implementation
- Module competition for workspace access
- Broadcasting conscious content to all modules
- Consciousness threshold detection

**Metacognitive Monitor**
- Confidence estimation for attention decisions
- Error detection in attention choices
- Performance history tracking
- Strategy revision recommendations

**Meta-Attention Controller**
- Selects among 8 attention strategies
- Tracks performance of each strategy
- Context-dependent strategy selection
- Adaptive strategy learning

### Advanced Cognitive Features

**Predictive World Model**
- Forward model: predicts next state from current state and action
- Inverse model: infers action from state transitions
- Reward model: estimates value of states
- Imagination-based planning
- Trajectory rollout for future simulation

**Curiosity Module**
- Intrinsic motivation through prediction error
- Novelty detection relative to observation history
- Empowerment estimation
- Combined intrinsic reward signal

**Attention Chunking**
- Groups related items using similarity
- Gestalt-like perceptual organization
- Reduces cognitive load
- Chunk-based attention distribution

**Goal Decomposition**
- Breaks complex goals into subgoals
- Hierarchical goal structure
- Subgoal ordering
- Sequential goal achievement

**Attention Persistence**
- Momentum for smooth attention transitions
- Inhibition of return to prevent attention loops
- History-based modulation
- Temporal coherence in attention


### Precision and Dynamics

**Hierarchical Precision Control**
- Multi-level precision learning
- Error-based precision adjustment
- Cross-level precision propagation
- Attention-modulated precision

**Learned Attractor Dynamics**
- Attractor landscape learning
- Multi-attractor competition
- Adaptive time constants
- Exploration noise injection
- Smooth attention transitions

---

## SECTION 2: FUNCTION BREAKDOWN (TECHNICAL)

### Tensor Helper Functions (Lines 20-73)

**tensor_exp** (Lines 24-30)
- Exponential function with gradient support
- Clips input to prevent overflow (-20 to 20)
- Backward pass: gradient = output * upstream_gradient
- Used for: probability computations, precision calculations

**tensor_log** (Lines 32-38)
- Logarithm with gradient support
- Clips input to prevent log(0)
- Backward pass: gradient = (1/input) * upstream_gradient
- Used for: entropy calculations, KL divergence

**tensor_sqrt** (Lines 40-46)
- Square root with gradient support
- Clips to non-negative values
- Backward pass: gradient = 0.5/sqrt(input) * upstream_gradient
- Used for: distance calculations, normalization

**tensor_sigmoid** (Lines 48-54)
- Sigmoid activation with gradient support
- Clips input to prevent overflow
- Backward pass: gradient = sigmoid * (1-sigmoid) * upstream_gradient
- Used for: gating, probability estimation

**tensor_abs** (Lines 56-62)
- Absolute value with gradient support
- Backward pass: gradient = sign(input) * upstream_gradient
- Used for: surprise computation, salience


### VariationalPosterior Class (Lines 78-143)

**__init__** (Lines 82-96)
- Initializes amortized inference network (encoder)
- Creates mode weights for multi-modal distributions
- Initializes sufficient statistics (means, log_vars) per mode
- Supports multiple posterior modes for complex distributions

**encode** (Lines 98-103)
- Amortized inference: maps observation to posterior parameters
- Returns mean and log_variance
- Enables fast inference without iterative optimization

**sample** (Lines 105-115)
- Reparameterization trick for gradient flow
- z = μ + σ * ε where ε ~ N(0,1)
- Allows backpropagation through sampling
- Can use encoded or stored parameters

**kl_divergence** (Lines 117-129)
- Computes KL[q(s|o) || p(s)] with proper gradient flow
- Formula: 0.5 * sum(σ²_p/σ²_q + (μ_q - μ_p)²/σ²_p - 1 + log(σ²_p/σ²_q))
- Used in variational free energy computation
- Measures complexity cost in active inference

**entropy** (Lines 131-134)
- Differential entropy of Gaussian: 0.5 * log(2πeσ²)
- Used for epistemic value computation
- Measures uncertainty in posterior

**parameters** (Lines 136-143)
- Returns all learnable parameters
- Includes encoder network, means, log_vars, mode_logits
- Used for gradient-based optimization

### LearnedPrecisionNetwork Class (Lines 146-180)

**__init__** (Lines 150-156)
- Initializes precision network (MLP)
- Base log_precision parameter
- Error history buffer (max 100 entries)
- Dynamically adjusts precision based on context

**forward** (Lines 158-172)
- Computes context-dependent precision
- Input: observation + prediction error
- Learned modulation added to base precision
- Updates error history for adaptation

**get_uncertainty** (Lines 174-177)
- Returns inverse of precision
- Uncertainty = 1 / (precision + ε)
- Used for epistemic value computation

**parameters** (Lines 179-180)
- Returns precision network parameters and base log_precision
- Used for learning optimal precision values


### ExpectedFreeEnergyComputer Class (Lines 183-270)

**__init__** (Lines 187-200)
- Transition model: predicts next state from (state, action)
- Observation model: predicts observation from state
- Preference model: estimates reward
- Uncertainty network for epistemic value
- All models output mean + log_variance

**compute_efe** (Lines 202-234)
- Computes Expected Free Energy for action selection
- EFE = -Epistemic_Value - Pragmatic_Value
- Epistemic: information gain (entropy reduction)
- Pragmatic: expected goal achievement
- Returns scalar EFE to minimize

**_compute_entropy** (Lines 236-240)
- Differential entropy: 0.5 * log(2πe * σ²)
- Used for epistemic value calculation
- Measures uncertainty in predictions

**select_attention_action** (Lines 242-258)
- Selects action that minimizes EFE
- Tests multiple candidate actions
- Returns best action (lowest EFE)
- Implements active inference action selection

**parameters** (Lines 260-265)
- Returns all model parameters
- Transition, observation, preference networks
- Used for end-to-end learning

### ActiveInferenceEngine Class (Lines 273-398)

**__init__** (Lines 277-290)
- Initializes variational posterior (3 modes)
- Precision network for weighted errors
- EFE computer for action selection
- Hierarchical belief levels (3 levels)
- Adaptive learning rate controller

**compute_vfe** (Lines 292-307)
- Variational Free Energy = Accuracy + Complexity
- Accuracy: precision-weighted prediction error
- Complexity: KL divergence from prior
- Computed at specified hierarchical level

**minimize_vfe** (Lines 309-349)
- Perception as inference
- Adaptive learning rate based on gradient magnitude
- Convergence detection (threshold 1e-4)
- Iterative optimization (max 10 steps)
- Updates posterior parameters

**hierarchical_belief_update** (Lines 351-373)
- Updates beliefs at 3 hierarchical levels
- Each level uses previous level as prior
- Gradient-based parameter updates
- Returns list of beliefs at each level

**active_inference_step** (Lines 375-387)
- Full active inference cycle
- Perception: minimize VFE
- Action: select attention minimizing EFE
- Returns attention action and VFE

**parameters** (Lines 389-394)
- Returns all learnable parameters
- Posterior, precision, EFE computer
- Used for joint optimization


### LearnedSymbolicRuleExtractor Class (Lines 403-625)

**__init__** (Lines 407-429)
- Rule encoder: maps (obs, attention) to rule embedding
- Rule library with hierarchical organization
- Three levels: primitive, composite, abstract
- Applicability network tests rule relevance
- Composition and abstraction networks

**extract_rule** (Lines 431-471)
- Extracts symbolic rule from experience
- Creates rule with condition, action, confidence
- Adds to appropriate hierarchy level
- Replaces least successful rule if library full

**compose_rules** (Lines 473-495)
- Combines two rules into composite rule
- Uses composition network
- Creates sequence of actions
- Tracks parent rules

**abstract_rules** (Lines 497-517)
- Abstracts multiple rules into strategy
- Combines embeddings from up to 3 rules
- Creates high-level pattern
- Generalizes across contexts

**generalize_rule** (Lines 519-532)
- Generalizes rule across similar contexts
- Averages embeddings
- Boosts confidence for generalized rules
- Marks as generalized type

**_extract_condition** (Lines 534-542)
- Extracts symbolic condition from observation
- Finds top-k salient features
- Returns feature pattern with indices and values

**_extract_action** (Lines 544-550)
- Extracts symbolic action from attention
- Identifies target location
- Returns attend_to action with strength

**apply_rules** (Lines 552-585)
- Applies learned rules to generate attention
- Tests applicability of each rule
- Weighted voting based on confidence
- Updates usage count

**parameters** (Lines 587-591)
- Returns rule encoder and applicability network
- Used for learning rule representations


### ProbabilisticLogicIntegrator Class (Lines 594-633)

**__init__** (Lines 598-600)
- Initializes fuzzy rule list
- Integrates probabilistic logic with neural attention

**add_fuzzy_rule** (Lines 602-608)
- Adds fuzzy rule: IF condition THEN consequence
- Condition returns truth value [0,1]
- Consequence returns attention tensor
- Confidence weight for rule

**evaluate** (Lines 610-633)
- Evaluates all fuzzy rules
- Aggregates results weighted by truth values
- Returns combined attention distribution
- Implements fuzzy logic inference

### NeuroSymbolicAttentionBridge Class (Lines 636-748)

**__init__** (Lines 640-651)
- Integrates grounding mechanism and reasoning engine
- Rule extractor for learning
- Logic integrator for fuzzy rules
- Bidirectional grounding networks

**concept_driven_attention** (Lines 653-662)
- Computes attention based on symbolic concept
- Grounds concept to neural space
- Returns attention distribution

**neural_to_symbolic_grounding** (Lines 664-675)
- Abstracts neural observation to symbolic concepts
- Uses neural_to_symbolic network
- Returns list of matched concepts

**symbolic_to_neural_grounding** (Lines 677-687)
- Grounds symbolic concepts to neural attention
- Encodes concepts as vector
- Returns normalized attention distribution

**forward** (Lines 689-710)
- Generates attention using neuro-symbolic reasoning
- Extracts concepts from observation
- Applies learned rules
- Applies fuzzy logic
- Combines all sources (40% rules, 30% logic, 30% symbolic)

**learn_from_experience** (Lines 712-716)
- Learns new rules from successful patterns
- Threshold: outcome > 0.7
- Extracts rule with concept

**parameters** (Lines 718-723)
- Returns all learnable parameters
- Rule extractor, grounding networks


### MultiTimescalePredictiveModule Class (Lines 751-831)

**__init__** (Lines 755-770)
- Separate predictor for each timescale (1, 5, 20 steps)
- Uncertainty estimators per timescale
- History buffer for higher-order dynamics
- Max history = max(timescales) + 10

**forward** (Lines 772-809)
- Predicts at multiple timescales
- Computes velocity and acceleration
- Returns (prediction, surprise, uncertainty) per timescale
- Requires at least 3 history entries

**get_integrated_prediction** (Lines 811-831)
- Integrates predictions across timescales
- Weighted combination (default: equal weights)
- Returns integrated prediction and surprise
- Used for unified prediction signal

**parameters** (Lines 833-839)
- Returns all predictor and uncertainty network parameters
- Used for multi-timescale learning

### HierarchicalPrecisionController Class (Lines 846-914)

**__init__** (Lines 850-868)
- Precision networks for each level (default 3)
- Base log-precision parameters
- Cross-level propagation networks
- Implements hierarchical precision learning

**forward** (Lines 870-900)
- Updates precision at all levels
- Context: error + attention
- Learned modulation + base precision
- Cross-level propagation
- Attention modulation (higher attention → higher precision)

**get_effective_precision** (Lines 902-904)
- Returns current precision at specified level
- Exponential of log_precision

**parameters** (Lines 906-914)
- Returns all precision networks, base precisions, propagation networks
- Used for hierarchical precision learning


### LearnedAttractorDynamics Class (Lines 917-1001)

**__init__** (Lines 921-938)
- Landscape network for potential gradient
- Learnable attractor centers (5 default)
- Adaptive time constant network
- Exploration noise parameter
- State tracking

**forward** (Lines 940-978)
- Updates attention using attractor dynamics
- Computes potential gradient (force)
- Attractor forces pull toward centers
- Adaptive time constant
- Dynamics: dx/dt = (-∇V + salience - x)/τ + noise
- Returns normalized attention distribution

**reset** (Lines 980-982)
- Resets dynamics state to uniform
- Used for new episodes

**parameters** (Lines 984-990)
- Returns landscape network, attractor centers, tau network
- Used for learning attractor landscape

### ObjectSegmentationModule Class (Lines 998-1056)

**__init__** (Lines 1002-1010)
- Object proposal network
- Feature extractor network
- Segments observations into objects

**forward** (Lines 1012-1042)
- Generates object proposals (centers, scales)
- Creates soft masks (Gaussian around center)
- Extracts features for each object
- Returns list of object dictionaries

**_create_mask** (Lines 1044-1050)
- Creates Gaussian mask around center
- Normalized to sum to 1
- Used for soft object segmentation

**parameters** (Lines 1052-1056)
- Returns proposal and feature networks
- Used for learning object segmentation


### ObjectTracker Class (Lines 1059-1213)

**__init__** (Lines 1063-1074)
- Tracks up to max_objects
- Maintains tracked and occluded objects
- Predicted positions during occlusion
- Object affordances learning

**predict_position** (Lines 1076-1082)
- Predicts object position using velocity
- Linear prediction: center + velocity
- Used during occlusion

**update** (Lines 1084-1177)
- Matches current objects to tracked objects
- Feature similarity + spatial proximity
- Re-identification of occluded objects
- Maintains objects during occlusion (up to 10 frames)
- Predictive tracking during occlusion
- Returns list of tracked objects (including predicted)

**learn_affordance** (Lines 1179-1183)
- Learns what can be done with object
- Stores affordance list per object

**get_affordances** (Lines 1185-1187)
- Returns learned affordances for object
- Used for action planning

### ObjectBasedAttention Class (Lines 1190-1243)

**__init__** (Lines 1194-1201)
- Segmentation module
- Object tracker
- Salience network for objects

**forward** (Lines 1203-1230)
- Segments observation into objects
- Tracks objects over time
- Computes salience for each object
- Creates attention map weighted by object masks
- Returns normalized attention distribution

**get_attended_object** (Lines 1232-1243)
- Returns object with highest attention overlap
- Used for identifying focus of attention

**parameters** (Lines 1245-1249)
- Returns segmentation and salience network parameters
- Used for learning object-based attention


### AttentionMemory Class (Lines 1256-1408)

**__init__** (Lines 1260-1275)
- Episode buffer (capacity 1000)
- Consolidated schemas
- Hierarchical organization: episodic, semantic, procedural
- Consolidation threshold (every 50 episodes)

**store** (Lines 1277-1298)
- Stores attention episode
- Automatic consolidation trigger
- Importance-based removal when full
- Updates timestamp

**consolidate_memories** (Lines 1300-1311)
- Clusters similar episodes
- Extracts schemas from clusters
- Adds to semantic memory
- Runs on recent 100 episodes

**_cluster_episodes** (Lines 1313-1343)
- Clusters similar episodes
- Similarity: 60% observation + 40% attention
- Threshold: 0.7
- Returns list of clusters

**_extract_schema** (Lines 1345-1361)
- Extracts common pattern from cluster
- Averages observations and attentions
- Computes expected outcome and variance
- Returns schema dictionary

**retrieve_similar** (Lines 1363-1388)
- Retrieves k most similar episodes/schemas
- Cosine similarity metric
- Updates access count
- Returns sorted by similarity

**extract_procedural_memory** (Lines 1390-1398)
- Extracts skills from successful sequences
- Requires multi-step sequences (≥3 steps)
- Stores in procedural memory

**retrieve_successful** (Lines 1400-1403)
- Returns successful episodes (outcome > threshold)
- Most recent k episodes

**get_statistics** (Lines 1405-1415)
- Returns memory statistics
- Size, mean/std/max/min outcomes


### WorkingMemoryIntegration Class (Lines 1418-1481)

**__init__** (Lines 1422-1433)
- 7 working memory slots (Miller's magic number)
- Gating network decides what to store
- Update network computes slot updates
- Age tracking per slot

**forward** (Lines 1435-1463)
- Updates working memory based on attention
- Gating: which slots to update (threshold 0.5)
- Update: blend old and new (90% old, 10% new)
- Age-based decay for unused slots
- WM-modulated attention output

**get_wm_content** (Lines 1465-1467)
- Returns current working memory content
- List of 7 slot tensors

**clear_wm** (Lines 1469-1473)
- Clears all working memory slots
- Resets ages to 0

**parameters** (Lines 1475-1479)
- Returns gate and update network parameters
- Used for learning WM dynamics

### CausalGraphLearner Class (Lines 1488-1720)

**__init__** (Lines 1492-1509)
- Learnable adjacency matrix
- Node embeddings (32-dim)
- Causal mechanism networks
- Conditional independence tester
- Intervention and observational data history

**test_conditional_independence** (Lines 1511-1525)
- Tests if X ⊥ Y | Z
- Uses independence network
- Returns boolean (independent if score > 0.5)

**pc_algorithm** (Lines 1527-1553)
- Peter-Clark algorithm for causal discovery
- Starts with complete graph
- Removes edges based on conditional independence
- Returns adjacency matrix

**discover_structure** (Lines 1555-1593)
- Discovers causal structure from data
- Uses PC algorithm on observational data
- Refines with interventional data
- Updates adjacency matrix (70% old, 30% new)

**intervene** (Lines 1595-1633)
- Performs intervention: do(X = value)
- Sets intervened node to value
- Propagates through causal graph
- Only downstream nodes affected
- Records intervention

**forward** (Lines 1635-1653)
- Predicts causal effects
- Stores observational data
- Aggregates causal parents
- Returns causal features

**parameters** (Lines 1655-1661)
- Returns adjacency, node embeddings, mechanism networks
- Used for learning causal structure


### CounterfactualReasoner Class (Lines 1664-1720)

**__init__** (Lines 1668-1669)
- Takes causal graph learner
- Enables counterfactual queries

**counterfactual_query** (Lines 1671-1697)
- Three-step algorithm: abduction, action, prediction
- Step 1: Infer latent state from observation
- Step 2: Apply intervention
- Step 3: Predict outcome under intervention
- Returns factual, counterfactual outcomes and causal effect

**explain_attention** (Lines 1699-1720)
- Explains attention using counterfactuals
- Tests alternative attention patterns
- Computes causal effects
- Returns top 5 explanations sorted by effect

### AttentionProgramSynthesizer Class (Lines 1727-1838)

**__init__** (Lines 1731-1750)
- Program library: scan, focus, track, compare, search
- Program encoder and decoder
- Learned programs storage

**_scan** (Lines 1752-1754)
- Uniform scanning attention
- Returns equal distribution

**_focus** (Lines 1756-1761)
- Focus on specific location
- Defaults to max activation

**_track** (Lines 1763-1767)
- Track moving target using gradient
- Attends to highest gradient

**_compare** (Lines 1769-1775)
- Compare multiple locations
- Attends to high variance regions

**_search** (Lines 1777-1782)
- Search for salient features
- Attends to outliers (deviation from mean)

**synthesize_program** (Lines 1784-1814)
- Synthesizes program from demonstrations
- Encodes demonstrations
- Averages embeddings
- Decodes to program sequence
- Returns top-k primitives

**execute_program** (Lines 1816-1826)
- Executes program sequence
- Combines primitive attentions
- Returns normalized attention

**parameters** (Lines 1828-1832)
- Returns encoder and decoder parameters
- Used for learning program synthesis


### TheoryOfMindModule Class (Lines 1845-1987)

**__init__** (Lines 1849-1869)
- Belief encoder and attention predictor
- Recursive reasoning network
- Belief divergence tracker
- Deception detector
- Common ground and private knowledge tracking

**infer_belief** (Lines 1871-1874)
- Infers agent's belief from observation
- Returns belief state embedding

**compute_belief_divergence** (Lines 1876-1880)
- Computes belief difference between agents
- Returns divergence score [0,1]

**detect_deception** (Lines 1882-1893)
- Detects if agent is deceptive
- Compares action to predicted action
- Uses deception network
- Returns deception probability

**update_common_ground** (Lines 1895-1902)
- Updates shared knowledge
- Tracks private vs common knowledge

**is_common_knowledge** (Lines 1904-1906)
- Checks if concept is shared
- Returns boolean

**perspective_taking** (Lines 1908-1920)
- Takes other agent's perspective
- Computes unique observations for each agent
- Returns (my_unique, other_unique)

**predict_attention** (Lines 1922-1926)
- Predicts agent's attention from belief
- Uses attention predictor network

**recursive_reasoning** (Lines 1928-1955)
- Recursive theory of mind
- Depth 1: What does other attend to?
- Depth 2: What does other think I attend to?
- Depth 3: What does other think I think they attend to?
- Max recursion: 3 levels

**parameters** (Lines 1957-1962)
- Returns belief encoder, attention predictor, recursive network
- Used for learning theory of mind


### MultiAgentAttentionCoordinator Class (Lines 1965-2023)

**__init__** (Lines 1969-1982)
- Theory of mind module
- Message encoder/decoder for communication
- Coordination network
- Adaptive norms for coordination and messages

**coordinate** (Lines 1984-2003)
- Coordinates attention with other agents
- Predicts others' attention using ToM (depth 2)
- Combines all observations
- Modulates by others' attention (20% weight)
- Returns coordinated attention

**communicate** (Lines 2005-2008)
- Encodes attention as message
- Returns message tensor

**receive_message** (Lines 2010-2013)
- Decodes message from other agent
- Returns attention tensor

**parameters** (Lines 2015-2021)
- Returns ToM, message encoder/decoder, coordination network
- Used for learning multi-agent coordination

### PredictiveWorldModel Class (Lines 2143-2213)

**__init__** (Lines 2147-2159)
- Forward model: (s,a) → s'
- Inverse model: (s,s') → a
- Reward model: s → r
- Uncertainty model for predictions

**imagine_trajectory** (Lines 2161-2170)
- Rolls out imagined future trajectory
- Applies actions sequentially
- Returns list of predicted states

**plan_with_imagination** (Lines 2172-2191)
- Plans action sequence using imagination
- Samples random action sequences
- Evaluates trajectories
- Returns best action sequence

**parameters** (Lines 2193-2199)
- Returns all model parameters
- Used for learning world model


### HierarchicalAttentionController Class (Lines 2202-2283)

**__init__** (Lines 2206-2225)
- Attention networks for each level (3 default)
- Top-down modulation networks
- Bottom-up salience networks
- Level-specific adaptive norms

**forward** (Lines 2227-2259)
- Computes hierarchical attention
- Bottom-up pass: propagates salience upward
- Top-down pass: propagates goals downward
- Integration at each level
- Returns dict {level: attention}

**parameters** (Lines 2261-2271)
- Returns all level attention, top-down, bottom-up networks
- Used for learning hierarchical attention

### CuriosityModule Class (Lines 2274-2337)

**__init__** (Lines 2278-2290)
- Prediction network for curiosity
- Novelty detector
- Empowerment estimator
- Observation history (max 1000)

**compute_curiosity** (Lines 2292-2296)
- Curiosity = prediction error
- Squared difference between predicted and actual

**compute_novelty** (Lines 2298-2315)
- Novelty relative to past observations
- Compares to last 100 observations
- Novelty = 1 - max_similarity
- Updates history

**compute_empowerment** (Lines 2317-2319)
- Estimates control over environment
- Mutual information between actions and states

**intrinsic_reward** (Lines 2321-2329)
- Combines curiosity, novelty, empowerment
- Weights: 40% curiosity, 30% novelty, 30% empowerment
- Returns total intrinsic motivation

**parameters** (Lines 2331-2336)
- Returns predictor, novelty, empowerment networks
- Used for learning curiosity


### AttentionChunking Class (Lines 2339-2403)

**__init__** (Lines 2343-2350)
- Chunk encoder network
- Similarity network for grouping
- Max chunks: 5

**chunk_by_similarity** (Lines 2352-2383)
- Groups indices by similarity
- Encodes each position
- Clusters using similarity threshold (0.7)
- Returns list of chunks

**attend_to_chunks** (Lines 2385-2401)
- Applies chunking to attention
- Averages attention within chunks
- Applies to all chunk members
- Returns normalized chunked attention

**parameters** (Lines 2403-2407)
- Returns chunk encoder and similarity network
- Used for learning chunking

### MetacognitiveMonitor Class (Lines 2410-2458)

**__init__** (Lines 2414-2421)
- Confidence estimator network
- Error detector network
- Performance history tracking

**estimate_confidence** (Lines 2423-2427)
- Estimates confidence in attention decision
- Input: observation + attention
- Returns confidence score [0,1]

**detect_error** (Lines 2429-2433)
- Detects if attention was erroneous
- Input: observation + attention + outcome
- Returns error probability

**should_revise_strategy** (Lines 2435-2443)
- Determines if strategy should be revised
- Compares recent to overall performance
- Revise if recent < 80% of overall

**update_performance** (Lines 2445-2449)
- Updates performance history
- Keeps last 100 entries

**parameters** (Lines 2451-2455)
- Returns confidence and error detector networks
- Used for learning metacognition


### GoalDecomposer Class (Lines 2461-2497)

**__init__** (Lines 2465-2472)
- Subgoal generator network
- Subgoal ordering network
- Max subgoals: 5

**decompose** (Lines 2474-2491)
- Decomposes goal into subgoals
- Generates subgoal vectors
- Orders subgoals by importance
- Returns ordered list of subgoals

**parameters** (Lines 2493-2497)
- Returns subgoal and ordering networks
- Used for learning goal decomposition

### AttentionPersistence Class (Lines 2500-2558)

**__init__** (Lines 2504-2516)
- Momentum network for smooth transitions
- Inhibition network for inhibition of return
- Attention history (max 10)
- Momentum coefficient: 0.7
- Inhibition coefficient: 0.3

**forward** (Lines 2518-2548)
- Applies momentum and inhibition
- Momentum: blends with previous attention
- Inhibition: reduces attention to recently attended
- Ensures non-negative values
- Returns normalized attention

**parameters** (Lines 2550-2555)
- Returns momentum and inhibition networks
- Used for learning persistence

### IntegratedInformationComputer Class (Lines 2566-2598)

**__init__** (Lines 2570-2571)
- Initializes with dimension
- Computes Φ (integrated information)

**compute_phi** (Lines 2573-2589)
- Computes integrated information Φ
- System entropy - sum of part entropies
- Measures irreducibility of causal structure
- Returns Φ ≥ 0

**_entropy** (Lines 2591-2596)
- Computes entropy of data
- Normalizes to probability distribution
- Returns -Σ p*log(p)

**is_conscious** (Lines 2598-2600)
- Determines if system is conscious
- Threshold: Φ > 0.5


### GlobalWorkspace Class (Lines 2603-2683)

**__init__** (Lines 2607-2627)
- Workspace buffer (conscious content)
- Module encoders and decoders
- Competition network for workspace access
- Broadcast strength parameter

**compete_for_workspace** (Lines 2629-2641)
- Modules compete for workspace access
- Concatenates all module inputs
- Computes competition scores
- Winner-take-all selection

**update_workspace** (Lines 2643-2649)
- Updates workspace with attended content
- Attention-weighted update
- Decay: 90% old, 10% new

**broadcast** (Lines 2651-2659)
- Broadcasts workspace to all modules
- Each module gets decoded broadcast
- Scaled by broadcast strength

**get_conscious_content** (Lines 2661-2663)
- Returns current workspace state
- Represents conscious content

**clear_workspace** (Lines 2665-2667)
- Clears global workspace
- Resets to zeros

**parameters** (Lines 2669-2677)
- Returns encoders, decoders, competition network
- Used for learning global workspace

### ConsciousnessSubstrate Class (Lines 2680-2745)

**__init__** (Lines 2684-2695)
- IIT computer for Φ
- Global workspace
- Consciousness threshold (0.5)
- History tracking

**process** (Lines 2697-2729)
- Processes attention through consciousness
- Computes integrated information (Φ)
- Determines if conscious
- Updates workspace if conscious
- Broadcasts to modules
- Returns consciousness metrics

**get_consciousness_report** (Lines 2731-2748)
- Generates consciousness report
- Current, mean, max Φ
- Consciousness ratio
- Total samples
- Currently conscious status


### MetaAttentionController Class (Lines 2755-2838)

**__init__** (Lines 2759-2779)
- Strategy selector network
- Adaptive norm for strategy selection
- Strategy embeddings (8 strategies)
- Performance history per strategy
- Strategy names: bottom_up, top_down, predictive, exploratory, exploitative, balanced, object_based, symbolic

**select_strategy** (Lines 2781-2795)
- Selects attention strategy based on context
- Input: observation + goal
- Computes strategy scores
- Returns strategy index and probabilities

**update_performance** (Lines 2797-2803)
- Updates strategy performance
- Keeps last 100 outcomes per strategy

**get_best_strategy** (Lines 2805-2811)
- Returns strategy with best average performance
- Used for strategy selection

**get_strategy_report** (Lines 2813-2824)
- Generates performance report
- Mean, std, count per strategy
- Returns dictionary

**parameters** (Lines 2826-2830)
- Returns strategy selector and embeddings
- Used for learning meta-attention

### AGIAttentionSubstrate Class (Lines 2845-3417)

**__init__** (Lines 2870-2970)
- Initializes all attention components
- Active inference engine
- Multi-timescale prediction
- Neuro-symbolic bridge
- Object-based attention
- Hierarchical precision
- Attractor dynamics
- Causal graph learner
- Program synthesizer
- Multi-agent coordinator
- Consciousness substrate
- Meta-attention controller
- Episodic memory
- Working memory
- 7 NEW modules: world model, hierarchical attention, curiosity, chunking, metacognition, goal decomposer, persistence
- Learnable salience weights (8 sources)


**forward** (Lines 2972-3178)
- Complete AGI-grade attention processing
- 23 processing steps:
  1. Goal decomposition
  2. Meta-attention strategy selection
  3. Hierarchical belief updating
  4. Active inference step
  5. Multi-timescale prediction
  6. Hierarchical attention
  7. Neuro-symbolic reasoning
  8. Object-based attention
  9. Curiosity-driven exploration
  10. Multi-agent coordination
  11. Dynamic routing with competition
  12. Attractor dynamics
  13. Attention chunking
  14. Persistence (momentum + inhibition)
  15. Working memory integration
  16. Metacognitive monitoring
  17. Hierarchical precision update
  18. Consciousness processing
  19. Causal discovery
  20. Episodic memory storage
  21. Rule learning and composition
  22. Meta-attention performance update
  23. Metacognitive performance update
- Returns comprehensive result dictionary with 15 fields

**explain_attention** (Lines 3180-3202)
- Comprehensive attention explanation
- Counterfactual explanations
- Strategy performance report
- Consciousness metrics
- Memory statistics
- Attended object identification

**synthesize_attention_program** (Lines 3204-3206)
- Synthesizes reusable program from demonstrations
- Delegates to program synthesizer

**execute_attention_program** (Lines 3208-3210)
- Executes synthesized program
- Delegates to program synthesizer

**integrate_with_core** (Lines 3212-3249)
- Deep integration with AGI core
- Wraps core.step with attention processing
- Modulates perception with attention
- Broadcasts conscious content
- Prints integration status

**parameters** (Lines 3251-3283)
- Returns all learnable parameters
- 17 component parameter lists
- Dynamic routing and competition parameters
- Salience weights

---

## SECTION 3: INTEGRATION METHODS

### Active Inference Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3004-3007)
**Purpose:** Selects attention actions that minimize Expected Free Energy
**Integration:**
- Takes hierarchical beliefs from belief updating (Line 2998-3001)
- Uses prior mean and log_var from goal
- Returns attention_action and VFE
- Attention action feeds into salience integration (Line 3032)

**Reuse Pattern:**
```python
# In other modules needing active inference
attention_action, vfe = self.active_inference.active_inference_step(
    obs, goal, prior_mean, prior_log_var
)
# Use attention_action as one salience source
```


### Multi-Timescale Prediction Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3009-3010)
**Purpose:** Provides predictions and surprise signals at multiple timescales
**Integration:**
- Called with current observation
- Returns predictions dict {timescale: (pred, surprise, uncertainty)}
- Integrated prediction and surprise extracted (Line 3010)
- Surprise feeds into salience integration (Line 3034)
- Prediction used for precision control (Line 3131)

**Reuse Pattern:**
```python
# In modules needing temporal prediction
predictions = self.predictive_module.forward(obs)
integrated_pred, integrated_surprise = self.predictive_module.get_integrated_prediction(obs)
# Use integrated_surprise for attention
# Use integrated_pred for error computation
```

### Neuro-Symbolic Bridge Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3015)
**Purpose:** Generates attention using symbolic reasoning and learned rules
**Integration:**
- Takes observation and knowledge base from context
- Returns symbolic_attention
- Feeds into salience integration (Line 3035)
- Learns from successful experiences (Line 3145-3151)
- Composes rules when enough primitives exist

**Reuse Pattern:**
```python
# In modules needing symbolic reasoning
symbolic_attention = self.symbolic_bridge.forward(obs, knowledge_base)
# Learn from experience
if outcome > threshold:
    self.symbolic_bridge.learn_from_experience(obs, attention, outcome)
```

### Object-Based Attention Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3018)
**Purpose:** Provides object-centric attention
**Integration:**
- Segments observation into objects
- Tracks objects over time with permanence
- Returns object_attention distribution
- Feeds into salience integration (Line 3036)
- Attended object retrieved for output (Line 3167)

**Reuse Pattern:**
```python
# In modules needing object tracking
object_attention = self.object_attention.forward(obs, goal)
attended_obj = self.object_attention.get_attended_object(attention)
# Use attended_obj for object-specific processing
```

### Causal Graph Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3139-3140)
**Purpose:** Discovers causal structure and enables interventions
**Integration:**
- Periodically discovers structure (10% probability)
- Stores observational data automatically
- Used by counterfactual reasoner for explanations
- Intervention method available for causal queries

**Reuse Pattern:**
```python
# In modules needing causal reasoning
# Discover structure
self.causal_graph.discover_structure([obs1, obs2, ...], interventions)
# Perform intervention
intervened_state = self.causal_graph.intervene(state, node_idx, value)
# Counterfactual query
cf_result = self.counterfactual_reasoner.counterfactual_query(
    obs, intervention, outcome_fn
)
```


### Theory of Mind Integration

**Used By:** MultiAgentAttentionCoordinator.coordinate (Line 1988-1992)
**Purpose:** Predicts other agents' attention and coordinates
**Integration:**
- Infers beliefs from observations
- Predicts attention using recursive reasoning (depth 2)
- Combines with own observation
- Modulates by others' actual attention
- Returns coordinated attention

**Reuse Pattern:**
```python
# In modules needing multi-agent reasoning
# Predict other's attention
other_belief = self.theory_of_mind.infer_belief(other_obs)
predicted_attn = self.theory_of_mind.predict_attention(other_belief)
# Recursive reasoning
recursive_attn = self.theory_of_mind.recursive_reasoning(
    my_obs, other_obs, depth=2
)
# Detect deception
is_deceptive = self.theory_of_mind.detect_deception(
    other_obs, other_action, predicted_action
)
```

### Consciousness Substrate Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3133-3135)
**Purpose:** Determines consciousness and broadcasts content
**Integration:**
- Processes observation and attention
- Computes integrated information (Φ)
- Determines consciousness state
- Updates global workspace if conscious
- Broadcasts to modules
- Returns consciousness metrics
- Used in core integration (Line 3226-3230)

**Reuse Pattern:**
```python
# In modules needing consciousness processing
module_inputs = [obs, goal, attention, prediction]
consciousness_result = self.consciousness.process(obs, attention, module_inputs)
if consciousness_result['is_conscious']:
    conscious_content = consciousness_result['conscious_content']
    # Use conscious content for enhanced processing
```

### Memory Systems Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3142-3143, 3145-3151)
**Purpose:** Stores experiences and retrieves similar patterns
**Integration:**
- Episodic memory stores every attention episode
- Automatic consolidation into schemas
- Retrieval for similar situations
- Procedural memory extraction from sequences
- Working memory modulates attention (Line 3128)

**Reuse Pattern:**
```python
# Store experience
self.episodic_memory.store(obs, attention, outcome, context)
# Retrieve similar
similar_episodes = self.episodic_memory.retrieve_similar(obs, k=5)
# Get statistics
stats = self.episodic_memory.get_statistics()
# Working memory integration
attention = self.working_memory.forward(obs, attention)
```


### Dynamic Routing Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3043-3118)
**Purpose:** Context-dependent weighting of attention sources
**Integration:**
- Creates routing context from obs + goal
- Dynamically creates routing network on first call
- Computes context-dependent weights
- Applies competitive dynamics between sources
- Non-linear gating for each source
- Winner-take-all sparse attention
- Replaces fixed weighted sum

**Reuse Pattern:**
```python
# In modules needing dynamic source integration
# Stack sources
sources = {'source1': tensor1, 'source2': tensor2, ...}
# Create routing context
routing_context = Tensor(np.concatenate([obs.data, goal.data]))
# Compute dynamic weights
routing_logits = self.routing_net(routing_context)
dynamic_weights = self.routing_norm(routing_logits)
# Apply competition
source_strengths = [s.sum().data for s in sources.values()]
# Compete and integrate
for idx, source in enumerate(sources.values()):
    gate = dynamic_weights[idx] * competed_strengths[idx]
    total += source * gate
```

### Hierarchical Attention Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3012-3013)
**Purpose:** Multi-level attention processing
**Integration:**
- Computes attention at 3 hierarchical levels
- Bottom-up salience propagation
- Top-down goal modulation
- Returns dict {level: attention}
- Level 0 attention feeds into salience integration (Line 3039)

**Reuse Pattern:**
```python
# In modules needing hierarchical processing
hierarchical_attentions = self.hierarchical_attention.forward(obs, goal)
# Use different levels for different purposes
scene_attention = hierarchical_attentions[0]  # Coarse
object_attention = hierarchical_attentions[1]  # Medium
feature_attention = hierarchical_attentions[2]  # Fine
```

### Curiosity Module Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3020-3025)
**Purpose:** Intrinsic motivation for exploration
**Integration:**
- Computes curiosity from prediction error
- Computes novelty from observation history
- Computes empowerment
- Returns combined intrinsic reward
- Feeds into salience integration (Line 3037)

**Reuse Pattern:**
```python
# In modules needing intrinsic motivation
if len(history) > 0:
    prev_obs = history[-1]
    intrinsic_reward = self.curiosity.intrinsic_reward(prev_obs, obs)
    # Use intrinsic_reward to bias exploration
```


### Metacognition Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3130, 3156)
**Purpose:** Monitors confidence and detects errors
**Integration:**
- Estimates confidence in attention decisions
- Tracks performance history
- Determines if strategy should be revised
- Updates performance after each episode
- Confidence used in result output (Line 3169)
- Revision flag in result output (Line 3171)

**Reuse Pattern:**
```python
# In modules needing metacognitive monitoring
confidence = self.metacognition.estimate_confidence(obs, attention)
should_revise = self.metacognition.should_revise_strategy()
if should_revise:
    # Switch to different strategy or approach
    pass
# Update after outcome
self.metacognition.update_performance(outcome)
```

### Goal Decomposition Integration

**Used By:** AGIAttentionSubstrate.forward (Line 2990-2992)
**Purpose:** Breaks complex goals into manageable subgoals
**Integration:**
- Decomposes goal at start of forward pass
- Uses first subgoal as current goal
- All subsequent processing uses current_goal
- Subgoals returned in result (Line 3166)

**Reuse Pattern:**
```python
# In modules needing hierarchical goal processing
subgoals = self.goal_decomposer.decompose(current_state, final_goal)
for subgoal in subgoals:
    # Process each subgoal sequentially
    result = process_with_subgoal(subgoal)
    if achieved(result, subgoal):
        continue
    else:
        break
```

### Attention Persistence Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3123-3127)
**Purpose:** Smooth attention transitions and inhibition of return
**Integration:**
- Applies momentum from previous attention
- Applies inhibition to recently attended locations
- Requires previous attention state
- Returns smoothed attention
- Stores current attention for next step

**Reuse Pattern:**
```python
# In modules needing temporal coherence
if hasattr(self, '_prev_attention'):
    attention = self.persistence.forward(self._prev_attention, new_attention)
else:
    attention = new_attention
self._prev_attention = attention
# Attention now has smooth transitions
```

### Program Synthesis Integration

**Used By:** AGIAttentionSubstrate methods (Line 3204-3210)
**Purpose:** Learn and execute reusable attention programs
**Integration:**
- Synthesizes programs from demonstrations
- Stores learned programs
- Executes programs on new observations
- Enables compositional attention routines

**Reuse Pattern:**
```python
# In modules needing program learning
# Collect demonstrations
demonstrations = [(obs1, attn1), (obs2, attn2), ...]
# Synthesize program
program = self.program_synthesizer.synthesize_program(demonstrations)
# Execute on new observation
attention = self.program_synthesizer.execute_program(program, new_obs)
```


### Meta-Attention Integration

**Used By:** AGIAttentionSubstrate.forward (Line 2995-2996, 3153)
**Purpose:** Selects and learns attention strategies
**Integration:**
- Selects strategy based on context at start
- Returns strategy index and probabilities
- Updates strategy performance after episode
- Strategy info in result (Line 3160-3161)

**Reuse Pattern:**
```python
# In modules needing strategy selection
strategy_idx, strategy_probs = self.meta_attention.select_strategy(obs, goal)
# Use strategy_idx to select processing mode
if strategy_idx == 0:  # bottom_up
    attention = bottom_up_attention(obs)
elif strategy_idx == 1:  # top_down
    attention = top_down_attention(obs, goal)
# ... other strategies
# Update after outcome
self.meta_attention.update_performance(strategy_idx, outcome)
```

### Precision Control Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3131-3132)
**Purpose:** Hierarchical precision weighting of errors
**Integration:**
- Takes list of prediction errors
- Takes current attention
- Updates precision at all levels
- Cross-level propagation
- Returns list of precisions per level
- Used in result output (Line 3164)

**Reuse Pattern:**
```python
# In modules needing hierarchical precision
errors = [obs - pred_level0, obs - pred_level1, obs - pred_level2]
precisions = self.precision_ctrl.forward(errors, attention)
# Use precisions to weight errors
weighted_errors = [err * prec for err, prec in zip(errors, precisions)]
```

### Attractor Dynamics Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3120)
**Purpose:** Smooth attention evolution with learned landscape
**Integration:**
- Takes total salience as input
- Applies attractor dynamics
- Temperature parameter controls exploration
- Returns smoothed attention
- Maintains internal state

**Reuse Pattern:**
```python
# In modules needing smooth dynamics
raw_attention = compute_raw_attention(obs)
smooth_attention = self.dynamics.forward(raw_attention, temperature=1.0)
# Reset for new episode
self.dynamics.reset()
```

### Chunking Integration

**Used By:** AGIAttentionSubstrate.forward (Line 3123)
**Purpose:** Groups related items for efficient processing
**Integration:**
- Takes observation and base attention
- Clusters by similarity
- Averages attention within chunks
- Returns chunked attention
- Reduces cognitive load

**Reuse Pattern:**
```python
# In modules needing perceptual grouping
raw_attention = compute_attention(obs)
chunked_attention = self.chunking.attend_to_chunks(obs, raw_attention)
# Chunked attention has grouped similar items
```

---

## USAGE EXAMPLES

### Basic Attention Processing
```python
from attention import AGIAttentionSubstrate
from nn import Tensor
import numpy as np

# Initialize
substrate = AGIAttentionSubstrate(dim=64, core=my_core)

# Process observation
obs = Tensor(np.random.randn(64))
goal = Tensor(np.random.randn(64))
result = substrate.forward(obs, goal)

# Access results
attention = result['attention']
confidence = result['confidence']
is_conscious = result['consciousness']['is_conscious']
```

### Multi-Agent Coordination
```python
# With other agents
other_obs = Tensor(np.random.randn(64))
other_attn = Tensor(np.ones(64) / 64)
result = substrate.forward(obs, goal, other_agents=[(other_obs, other_attn)])

# Access social attention
social_attention = result['social_attention']
```

### Explanation and Introspection
```python
# Explain attention decision
explanation = substrate.explain_attention(obs, attention)
counterfactuals = explanation['counterfactual_explanations']
strategy_perf = explanation['strategy_performance']
consciousness_metrics = explanation['consciousness_metrics']
```

### Program Learning
```python
# Learn from demonstrations
demos = [(obs1, attn1), (obs2, attn2), (obs3, attn3)]
program = substrate.synthesize_attention_program(demos)

# Execute learned program
new_attention = substrate.execute_attention_program(program, new_obs)
```

### Core Integration
```python
# Integrate with AGI core
substrate.integrate_with_core()

# Now core.step uses attention automatically
result = core.step(obs, goal)
```

---

## KEY DESIGN PRINCIPLES

1. **Gradient Flow**: All operations support backpropagation through custom backward functions
2. **Modularity**: Each component is independent and can be used separately
3. **Hierarchical Processing**: Multiple levels of abstraction (beliefs, attention, precision)
4. **Active Inference**: Minimizes surprise and Expected Free Energy
5. **Sparse Attention**: Winner-take-all creates focused attention (top 10%)
6. **Dynamic Routing**: Context-dependent weighting, not fixed combinations
7. **Competitive Dynamics**: Sources inhibit each other based on strength
8. **Memory Integration**: Episodic, semantic, procedural, and working memory
9. **Consciousness Metrics**: IIT and Global Workspace Theory
10. **Multi-Agent Capable**: Theory of mind and coordination

---

## PERFORMANCE CHARACTERISTICS

- **Attention Sparsity**: ~10% of inputs receive attention (configurable)
- **Memory Capacity**: 1000 episodes (configurable)
- **Working Memory**: 7 slots (Miller's magic number)
- **Hierarchical Levels**: 3 (beliefs, attention, precision)
- **Timescales**: 3 (short, medium, long-term)
- **Strategies**: 8 attention strategies
- **Consciousness Threshold**: Φ > 0.5
- **Rule Hierarchy**: 3 levels (primitive, composite, abstract)

---

**END OF DOCUMENTATION**
