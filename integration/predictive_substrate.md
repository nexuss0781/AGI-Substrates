# 🧠 PREDICTIVE SUBSTRATE DOCUMENTATION

## 📋 SECTION 1: FEATURE BREAKDOWN

### 🎯 Overview
The Predictive Substrate is an AGI-grade hierarchical predictive coding system that implements causal reasoning, active inference, object-centric representations, and deep memory integration. It serves as the core world modeling and planning engine for the AGI system.

### ✨ Core Features Implemented

#### 1️⃣ **Hierarchical Predictive Coding**
- Multi-layer predictive hierarchy with bottom-up and top-down processing
- Each layer learns compressed representations of the layer below
- Prediction error minimization drives learning and inference
- Sparse competition for efficient representation

#### 2️⃣ **Causal Reasoning Engine**
- Structural Causal Models (SCM) with learned mechanisms
- Do-calculus for interventional predictions
- Pearl's three-step counterfactual reasoning (Abduction-Action-Prediction)
- PC algorithm for causal structure discovery
- Gradient-based noise inference for counterfactuals

#### 3️⃣ **Active Inference & Goal-Directed Behavior**
- Expected Free Energy (EFE) minimization for action selection
- Pragmatic value (goal achievement) + Epistemic value (uncertainty reduction)
- Hierarchical policy library with attention-based retrieval
- Learned policy composition and evolution
- Intrinsic motivation system for exploration

#### 4️⃣ **Object-Centric World Model**
- Slot attention mechanism for object segmentation
- Object permanence tracking with Kalman filtering
- Learned physics constraints (gravity, friction, elasticity)
- Relational reasoning between objects
- Physics-constrained dynamics prediction


#### 5️⃣ **Memory Integration**
- Working Memory with VSA (Vector-Symbolic Architecture) binding
- Short-Term Memory with recency and importance tracking
- Long-Term Memory (episodic, semantic, procedural)
- Memory consolidation and retrieval mechanisms
- Cognitive routines for complex operations

#### 6️⃣ **Meta-Learning & Rapid Adaptation**
- MAML-style fast adaptation to new tasks
- Hypernetwork-based parameter generation
- Task-specific adaptation in few steps
- Support set gradient computation

#### 7️⃣ **Continual Learning**
- Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention
- Synaptic Intelligence (SI) tracking
- Experience replay with priority sampling
- Task-specific parameter protection

#### 8️⃣ **Uncertainty Quantification**
- Bayesian uncertainty estimation with variational inference
- Epistemic uncertainty (model/knowledge uncertainty - reducible)
- Aleatoric uncertainty (data noise - irreducible)
- Proper separation and tracking of both types

#### 9️⃣ **Hyperbolic Geometry for Concepts**
- Poincaré ball embeddings for hierarchical concepts
- Hyperbolic mesh substrate integration
- Natural hierarchy representation in hyperbolic space
- Concept discovery and pruning

#### 🔟 **Multi-Agent Theory of Mind**
- Bayesian Theory of Mind with particle filtering
- Goal and emotion inference from observations
- Nash equilibrium computation for game-theoretic planning
- Coalition formation and strategic communication
- Multi-agent simulation capabilities

#### 1️⃣1️⃣ **Symbolic Reasoning Integration**
- Neural-symbolic bridge for logical inference
- Symbolic fact extraction from latent representations
- Forward chaining and resolution-based reasoning
- Rule learning and knowledge base management
- Explainable decision-making

#### 1️⃣2️⃣ **Hierarchical Options Framework**
- Temporal abstraction with options (skills)
- Automatic skill discovery from trajectories
- Hierarchical RL with option policies
- Subgoal discovery using graph betweenness centrality

#### 1️⃣3️⃣ **Contrastive Self-Supervised Learning**
- InfoNCE loss for representation learning
- Positive/negative pair generation
- Contrastive encoder for embedding space
- Temperature-scaled similarity computation

#### 1️⃣4️⃣ **Causal Abstraction Hierarchy**
- Multi-level causal models (latent → objects → concepts)
- Abstraction functions between levels
- Intervention effects at different abstractions
- Cross-level causal reasoning

#### 1️⃣5️⃣ **Self-Modeling & Metacognition**
- Confidence calibration with Expected Calibration Error (ECE)
- Capability boundary detection
- Out-of-distribution detection
- Learning progress tracking
- Metacognitive awareness reporting

#### 1️⃣6️⃣ **Compositional Generalization**
- Component detection and recombination
- Disentanglement scoring
- Novel combination handling
- Systematic generalization testing

#### 1️⃣7️⃣ **World Model Imagination**
- Dreamer-style latent space rollouts
- Action-conditioned trajectory prediction
- Model-based planning without environment interaction
- Uncertainty-aware imagination

---

## 🔧 SECTION 2: FUNCTION BREAKDOWN

### 📦 Class: CausalPredictiveLayer (Lines 78-790)

#### 🎯 Purpose
Core predictive layer implementing causal reasoning, active inference, and uncertainty quantification.


#### 🔹 `__init__` (Lines 94-148)
**Location:** Lines 94-148  
**Technical Implementation:**
- Initializes latent state `z` with specified dimensions
- Creates epistemic variance (model uncertainty) and aleatoric variance (data noise)
- Sets up generative model `W` for z → x reconstruction
- Initializes observational dynamics matrix `A_obs` for temporal prediction
- Creates action influence matrix `B` for action effects
- Integrates CausalDiscoveryEngineV2 for structure learning
- Configures adaptive inference with momentum (Adam-style)
- Initializes VSA binding space for variable binding
- Sets learning rates, sparsity parameters, and convergence thresholds

#### 🔹 `infer` (Lines 150-232)
**Location:** Lines 150-232  
**Technical Implementation:**
- Performs hierarchical inference with interventional support
- Handles do-operator interventions via `_interventional_inference`
- Implements iterative inference with early stopping (convergence detection)
- Uses precision-weighted gradients (inverse variance weighting)
- Applies Adam-style momentum with bias correction
- Adaptive learning rate based on gradient magnitude
- Integrates top-down predictions from higher layers
- Applies sparsity constraints and sparse competition
- Updates latent usage statistics for tracking

#### 🔹 `_interventional_inference` (Lines 234-251)
**Location:** Lines 234-251  
**Technical Implementation:**
- Implements Pearl's do-calculus for interventional inference
- Removes incoming edges to intervened variables
- Fixes intervened variable values
- Infers remaining variables conditioned on intervention
- Returns modified latent state under intervention

#### 🔹 `_compute_inference_gradient` (Lines 253-272)
**Location:** Lines 253-272  
**Technical Implementation:**
- Computes precision-weighted inference gradient
- Matches precision dimensions to error dimensions
- Applies weighted error: `error * precision`
- Backpropagates through generative model W
- Returns gradient in latent space

#### 🔹 `_apply_sparse_competition` (Lines 274-341)
**Location:** Lines 274-341  
**Technical Implementation:**
- Learned lateral inhibition with dynamic competition network
- Competition network predicts inhibition weights from activations + context
- Context includes: mean, std, max, min, age, sparsity level
- Adaptive threshold based on learned weights (75th percentile)
- Lateral inhibition with learned receptive fields
- Distance-based decay for neighbor inhibition
- Ensures minimum sparsity (activates top-k if too sparse)
- Replaces fixed sigmoid with adaptive neural competition

#### 🔹 `learn` (Lines 343-383)
**Location:** Lines 343-383  
**Technical Implementation:**
- Computes reconstruction error from generative model
- Updates generative weights W via gradient descent
- Applies weight decay for regularization
- Learns observational dynamics A via `_learn_observational_dynamics`
- Learns action dynamics B via `_learn_action_dynamics`
- Updates uncertainty estimates (epistemic + aleatoric)
- Discovers causal structure every 50 steps
- Stores experience in memory system
- Returns reconstruction error magnitude

#### 🔹 `_learn_observational_dynamics` (Lines 385-398)
**Location:** Lines 385-398  
**Technical Implementation:**
- Learns temporal dynamics: z_t+1 = A @ z_t
- Predicts current z from previous z using A_obs
- Computes temporal prediction error
- Updates A matrix via gradient descent
- Clips weights to [-2.0, 2.0] for stability

#### 🔹 `_learn_action_dynamics` (Lines 400-416)
**Location:** Lines 400-416  
**Technical Implementation:**
- Learns action influence: Δz = B @ a
- Predicts passive dynamics (no action)
- Computes action effect as residual
- Updates B matrix based on action-effect correlation
- Clips weights for stability

#### 🔹 `_update_uncertainty` (Lines 418-450)
**Location:** Lines 418-450  
**Technical Implementation:**
- Bayesian uncertainty estimation with variational inference
- Uses BayesianUncertaintyEstimator with 50 samples
- Projects error to latent space if needed
- Estimates epistemic variance (knowledge uncertainty - reducible)
- Estimates aleatoric variance (data noise - irreducible)
- Blends observed error with estimated aleatoric variance
- Clips to reasonable bounds [0.01, 10.0] and [0.001, 10.0]

#### 🔹 `_discover_causal_structure` (Lines 452-497)
**Location:** Lines 452-497  
**Technical Implementation:**
- Uses PC algorithm for causal discovery (NO correlation fallback)
- Performs conditional independence tests
- Requires minimum 30 observations for statistical validity
- Discovers DAG structure every 50 steps
- Updates causal parents from discovered DAG
- Creates LearnedStructuralEquation for each variable
- Stores structural equations in causal_mechanisms dict


#### 🔹 `predict_next_latent` (Lines 499-514)
**Location:** Lines 499-514  
**Technical Implementation:**
- Predicts next latent state using temporal dynamics
- Applies observational dynamics: z_t+1 = A @ z_t
- Adds action influence if action provided: z_t+1 += B @ a_t
- Returns predicted next state tensor

#### 🔹 `predict_intervention` (Lines 516-547)
**Location:** Lines 516-547  
**Technical Implementation:**
- Implements do-calculus for interventional prediction
- Cuts incoming edges to intervened variables
- Fixes intervened variable values throughout trajectory
- Forward simulates for specified horizon
- Keeps intervened variables fixed at each step
- Returns trajectory of predicted states

#### 🔹 `counterfactual` (Lines 549-619)
**Location:** Lines 549-619  
**Technical Implementation:**
- Pearl's three-step counterfactual algorithm:
  1. **Abduction:** Infer exogenous noise via gradient-based optimization
  2. **Action:** Apply intervention (cut incoming edges)
  3. **Prediction:** Forward simulate with inferred noise
- Performs topological sort for causal ordering
- Uses learned structural equations with inferred noise
- Handles non-linear mechanisms through mechanism networks
- Returns counterfactual value at query dimension

#### 🔹 `_abduce_noise_gradient_based` (Lines 621-691)
**Location:** Lines 621-691  
**Technical Implementation:**
- Gradient-based optimization to infer exogenous noise
- Solves inverse problem: find noise that reconstructs evidence
- Initializes noise near zero (Gaussian * 0.1)
- Iterates 50 steps with learning rate 0.01
- Forward pass: reconstruct state from noise using SCM
- Computes reconstruction error for evidence dimensions
- Gradient descent on noise to minimize error
- Early stopping when error < 1e-6

#### 🔹 `_topological_sort` (Lines 693-724)
**Location:** Lines 693-724  
**Technical Implementation:**
- Kahn's algorithm for topological sorting
- Computes in-degrees for all variables
- Initializes queue with zero in-degree nodes
- Processes nodes in causal order
- Detects cycles (returns simple order if cycle found)
- Returns variables in causal ordering

#### 🔹 `_abduce_noise` (Lines 726-766)
**Location:** Lines 726-766  
**Technical Implementation:**
- Alternative noise inference using structural equations
- For each evidence dimension, computes expected value from parents
- Noise = observed_value - expected_value
- Uses structural equation sampling if available
- Falls back to linear approximation if needed
- Returns noise vector for all dimensions

#### 🔹 `bind_variable` (Lines 768-770)
**Location:** Lines 768-770  
**Technical Implementation:**
- VSA (Vector-Symbolic Architecture) binding
- Binds role to filler using circular convolution
- Returns bound representation

#### 🔹 `unbind_variable` (Lines 772-774)
**Location:** Lines 772-774  
**Technical Implementation:**
- VSA unbinding operation
- Recovers filler from bound vector given role
- Uses circular correlation (inverse of convolution)

#### 🔹 `get_uncertainty` (Lines 776-782)
**Location:** Lines 776-782  
**Technical Implementation:**
- Returns uncertainty decomposition dictionary
- Epistemic: model/knowledge uncertainty (reducible)
- Aleatoric: data noise (irreducible)
- Total: sum of both components

#### 🔹 `parameters` (Lines 784-790)
**Location:** Lines 784-790  
**Technical Implementation:**
- Returns all trainable parameters
- Includes W (generative), A_obs (dynamics), B (action) parameters
- Used for optimization and meta-learning

---

### 📦 Class: ObjectCentricPredictiveLayer (Lines 794-1110)

#### 🎯 Purpose
Object-centric world model with slot attention, physics simulation, and relational reasoning.

#### 🔹 `__init__` (Lines 805-843)
**Location:** Lines 805-843  
**Technical Implementation:**
- Initializes slot representations (object-centric)
- Creates AGIMultiHeadSelfAttention for slot attention
- Sets up object encoder (input → slot features)
- Sets up object decoder (slots → reconstruction)
- Initializes learned object dynamics network
- Creates relation network for object interactions
- Initializes physics network (predicts gravity, friction, mass, elasticity)
- Tracks object positions and velocities
- Uses learned physics instead of hardcoded parameters


#### 🔹 `segment_objects` (Lines 845-901)
**Location:** Lines 845-901  
**Technical Implementation:**
- Iterative slot attention refinement (3 iterations)
- Encodes input to slot features via object_encoder
- Applies multi-head self-attention for competitive binding
- Computes attention scores between slots (cosine similarity)
- Softmax competition: slots inhibit similar slots
- Applies inhibition based on competition weights
- Residual connections with blend factor (α=0.7)
- Normalizes slots after each iteration
- Returns refined slot representations

#### 🔹 `predict_object_dynamics` (Lines 903-954)
**Location:** Lines 903-954  
**Technical Implementation:**
- Two-pass prediction: relational influences + individual dynamics
- **Pass 1:** Computes pairwise relations between all objects
  - Relation network predicts influence of object j on object i
  - Accumulates relational influences (scaled by 0.1)
- **Pass 2:** Predicts next state for each object
  - Combines current slot with previous slot (temporal context)
  - Applies object dynamics network
  - Adds relational influences
  - Applies learned physics constraints
- Stores previous slots for next iteration

#### 🔹 `_apply_physics` (Lines 956-1021)
**Location:** Lines 956-1021  
**Technical Implementation:**
- Uses PhysicsConstrainedWorldModel for learned physics
- Physics network predicts: [gravity, friction, mass, elasticity]
- Applies learned gravity: velocity_y += (gravity/mass) * dt
- Applies learned friction: velocity *= friction
- Applies action effects if provided
- Updates positions: position += velocity * dt
- Boundary collisions with elasticity: velocity *= -elasticity
- Applies physics constraints (permanence, causality, energy conservation)
- Stores state for next iteration

#### 🔹 `track_objects` (Lines 1023-1084)
**Location:** Lines 1023-1084  
**Technical Implementation:**
- Uses ObjectPermanenceTracker with Kalman filtering
- Updates tracker with current slot observations
- Builds correspondence mapping via tracked object IDs
- Handles occluded objects:
  - Predicts future positions for occluded objects
  - Places predictions in free slots
- Maintains prev_slot_to_object_id mapping
- Returns correspondence dictionary (prev_idx → curr_idx)

#### 🔹 `reconstruct` (Lines 1086-1095)
**Location:** Lines 1086-1095  
**Technical Implementation:**
- Concatenates all slot representations
- Decodes through object_decoder network
- Returns reconstructed input

#### 🔹 `parameters` (Lines 1097-1110)
**Location:** Lines 1097-1110  
**Technical Implementation:**
- Returns all trainable parameters from:
  - Slot attention mechanism
  - Object encoder and decoder
  - Object dynamics network
  - Relation network
  - Physics network

---

### 📦 Class: ActiveInferencePredictiveAgent (Lines 1114-1381)

#### 🎯 Purpose
Active inference agent with Expected Free Energy minimization and hierarchical planning.

#### 🔹 `__init__` (Lines 1125-1162)
**Location:** Lines 1125-1162  
**Technical Implementation:**
- Stores reference to predictive layer
- Initializes preferences (desired observations)
- Creates LearnedHierarchicalPlanner (3 levels) if available
- Generates policy library (10 policies)
- Initializes IntrinsicMotivationSystem
- Tracks EFE and action history (1000 steps)

#### 🔹 `_generate_policies` (Lines 1164-1264)
**Location:** Lines 1164-1264  
**Technical Implementation:**
- Uses AttentionBasedPolicyLibrary (NO random fallback)
- Retrieves relevant policies from library (top-k=5)
- Generates action sequences from policy networks
- Composes existing policies for new policies
- Adds exploration noise (decaying with time)
- Creates bootstrap policies with learned structure
- Adds new policies to library for future use
- Returns list of action sequences (horizon length)

#### 🔹 `expected_free_energy` (Lines 1266-1308)
**Location:** Lines 1266-1308  
**Technical Implementation:**
- Simulates trajectory under policy
- Computes pragmatic value: distance to goal
- Computes epistemic value: mean epistemic uncertainty
- Combines with temporal discounting (γ=0.9)
- Adds action cost (L2 norm of actions)
- Returns total EFE (lower = better)

#### 🔹 `select_action` (Lines 1310-1340)
**Location:** Lines 1310-1340  
**Technical Implementation:**
- Evaluates EFE for all policies
- Applies softmax over negative EFE
- Numerical stability: normalize before exp
- Samples policy proportional to exp(-β * EFE)
- Returns first action (receding horizon control)
- Stores action in history

#### 🔹 `act_and_observe` (Lines 1342-1355)
**Location:** Lines 1342-1355  
**Technical Implementation:**
- Complete active inference cycle:
  1. Perception: infer latent state from observation
  2. Action selection: minimize EFE
  3. Return action
- Integrates perception and action


#### 🔹 `update_preferences` (Lines 1357-1359)
**Location:** Lines 1357-1359  
**Technical Implementation:**
- Updates goal state (preferences)
- Used for goal-directed behavior

#### 🔹 `get_intrinsic_motivation` (Lines 1361-1373)
**Location:** Lines 1361-1373  
**Technical Implementation:**
- Computes motivation from recent EFE
- High EFE = high uncertainty = high motivation
- Normalizes to [0, 1] range
- Returns motivation level

#### 🔹 `parameters` (Lines 1375-1381)
**Location:** Lines 1375-1381  
**Technical Implementation:**
- Returns predictive layer parameters
- Used for optimization

---

### 📦 Class: AGIPredictiveHierarchy (Lines 1385-3951)

#### 🎯 Purpose
Top-level hierarchical system integrating all components for AGI-grade prediction and reasoning.

#### 🔹 `__init__` (Lines 1395-1540)
**Location:** Lines 1395-1540  
**Technical Implementation:**
- Creates hierarchical layers (default: [32, 16, 8])
- Initializes CausalPredictiveLayer for each level
- Creates ObjectCentricPredictiveLayer (6 slots)
- Initializes ActiveInferencePredictiveAgent
- Sets up AGIMemorySystem (working, short-term, long-term)
- Creates HyperbolicMeshSubstrate for concept learning
- Initializes ContinualLearningSystem (EWC, SI, replay)
- Sets up MetaLearningController (MAML, hypernetworks)
- Creates IntrinsicMotivationSystem
- Initializes HierarchicalRLController for options
- Sets up AGIMultiHeadSelfAttention if available
- Creates BayesianTheoryOfMind if available
- Initializes SymbolicReasoner if available
- Sets up causal abstraction hierarchy (3 levels)
- Creates contrastive encoder for self-supervised learning
- Configures consolidation interval and buffers

#### 🔹 `process` (Lines 1542-1796)
**Location:** Lines 1542-1796  
**Technical Implementation:**
- **Step 1: Bottom-up inference**
  - Processes through each layer hierarchically
  - Each layer infers latent state from input
  - Passes latent to next layer as input
- **Step 2: Object segmentation**
  - Segments input into object slots
  - Predicts object dynamics if action provided
  - Reconstructs from object slots
- **Step 3: Memory retrieval**
  - Queries memory system with top latent
  - Retrieves relevant episodic memories
  - Synthesizes knowledge from retrieved memories
- **Step 4: Concept learning**
  - Identifies or creates concept in hyperbolic mesh
  - Updates concept embeddings
  - Tracks concept access patterns
- **Step 5: Advanced reasoning** (if enabled)
  - Hierarchical attention across layers
  - Theory of Mind for multi-agent reasoning
  - Symbolic reasoning with fact extraction
  - Forward chaining for logical inference
- **Step 6: Learning** (if enabled)
  - Updates generative models and dynamics
  - Contrastive learning step
  - Causal abstraction learning (every 50 steps)
  - Stores experience in memory
  - Continual learning updates
- **Step 7: Consolidation** (periodic)
  - Memory consolidation
  - Concept pruning
- **Step 8: Uncertainty quantification**
  - Collects uncertainty from all layers
- **Step 9: Results compilation**
  - Returns comprehensive results dictionary

#### 🔹 `_compute_surprise` (Lines 1798-1810)
**Location:** Lines 1798-1810  
**Technical Implementation:**
- Reconstructs from top latent
- Computes prediction error magnitude
- Returns surprise (free energy)

#### 🔹 `predict_intervention` (Lines 1812-1819)
**Location:** Lines 1812-1819  
**Technical Implementation:**
- Delegates to top layer's predict_intervention
- Returns trajectory under intervention

#### 🔹 `counterfactual_reasoning` (Lines 1821-1830)
**Location:** Lines 1821-1830  
**Technical Implementation:**
- Delegates to top layer's counterfactual reasoning
- Returns counterfactual value

#### 🔹 `imagine_trajectory` (Lines 1832-2025)
**Location:** Lines 1832-2025  
**Technical Implementation:**
- Dreamer-style latent imagination for model-based planning
- **With WorldModel:**
  - Converts latent to slots
  - Rolls out trajectory in world model
  - Action-conditioned predictions
  - Extracts uncertainty from predictions
- **Without WorldModel (fallback):**
  - Uses predictive layers for imagination
  - Predicts next latent with action
  - Decodes through hierarchy to observation space
  - Tracks uncertainty per step
- Computes rewards (distance to goal)
- Calculates trajectory value (discounted sum)
- Computes information gain (epistemic value)
- Returns imagined states, observations, uncertainties, rewards

#### 🔹 `set_goal` (Lines 2027-2029)
**Location:** Lines 2027-2029  
**Technical Implementation:**
- Sets goal for active inference agent
- Updates preferences

#### 🔹 `test_compositional_generalization` (Lines 2031-2162)
**Location:** Lines 2031-2162  
**Technical Implementation:**
- Tests compositional generalization capabilities
- **Component Detection:** Checks if expected components are present
  - Uses concept mesh similarity
  - Computes detection rate
- **Disentanglement Score:** Measures independence of latent dimensions
  - Variance-based metric
  - Normalized entropy of variances
- **Novel Combination Performance:** Tests on unseen combinations
  - Measures coherence (inverse uncertainty)
- **Systematic Generalization:** Tests systematic recombination
  - Measures consistency across similar components
  - Computes overlap vs similarity correlation
- Returns comprehensive metrics dictionary


#### 🔹 `model_own_capabilities` (Lines 2164-2326)
**Location:** Lines 2164-2326  
**Technical Implementation:**
- Self-modeling and metacognitive awareness
- **Confidence Calibration:**
  - Uses Expected Calibration Error (ECE) with 10 bins
  - Compares predicted confidence vs actual accuracy
  - Calibration score = 1 - ECE
- **Known Capabilities:**
  - Identifies concepts with high mastery (>0.7)
  - Tracks access count and mastery level
  - Sorts by mastery
- **Capability Boundaries:**
  - Identifies high epistemic uncertainty areas
  - Finds uncertain dimensions per layer
  - Marks knowledge boundaries
- **Out-of-Distribution Detection:**
  - Measures distance to nearest learned concept
  - Normalizes to [0, 1] score
  - Tracks OOD scores over time
- **Learning Progress:**
  - Compares recent vs older errors
  - Computes improvement rate
- Returns comprehensive self-model report with metacognitive awareness

#### 🔹 `plan_with_options` (Lines 2328-2406)
**Location:** Lines 2328-2406  
**Technical Implementation:**
- Hierarchical planning with temporal abstraction
- Discovers subgoals from trajectories
- Plans high-level option sequence
- Executes low-level actions within options
- Returns hierarchical plan with options and actions

#### 🔹 `discover_subgoals` (Lines 2408-2464)
**Location:** Lines 2408-2464  
**Technical Implementation:**
- Discovers subgoals using graph betweenness centrality
- Builds state graph from trajectory
- Computes betweenness centrality for each state
- Identifies bottleneck states (high centrality)
- Merges with reward-based subgoals
- Adds to subgoal library
- Returns discovered subgoal states

#### 🔹 `_build_state_graph` (Lines 2466-2503)
**Location:** Lines 2466-2503  
**Technical Implementation:**
- Builds graph from state trajectory
- Connects temporally adjacent states
- Connects similar states (cosine similarity > 0.8)
- Returns adjacency list representation

#### 🔹 `_compute_betweenness_centrality` (Lines 2505-2555)
**Location:** Lines 2505-2555  
**Technical Implementation:**
- Computes betweenness centrality for all nodes
- For each pair of nodes, counts paths through each intermediate node
- Normalizes by total number of paths
- Returns centrality scores

#### 🔹 `_count_paths_through_nodes` (Lines 2557-2600)
**Location:** Lines 2557-2600  
**Technical Implementation:**
- DFS-based path counting
- Counts how many paths from source to target pass through each node
- Uses visited set to avoid cycles
- Returns path counts per node

#### 🔹 `_merge_subgoals` (Lines 2602-2629)
**Location:** Lines 2602-2629  
**Technical Implementation:**
- Merges graph-based and reward-based subgoals
- Removes duplicates (similarity > 0.9)
- Limits to top 10 subgoals
- Returns merged subgoal list

#### 🔹 `_add_to_subgoal_library` (Lines 2631-2660)
**Location:** Lines 2631-2660  
**Technical Implementation:**
- Adds subgoal to library with metadata
- Extracts graph features
- Stores start state, goal state, subgoals
- Limits library size to 100 entries

#### 🔹 `_extract_graph_features` (Lines 2662-2700)
**Location:** Lines 2662-2700  
**Technical Implementation:**
- Extracts graph topology features:
  - Average degree
  - Degree standard deviation
  - Clustering coefficient
  - Graph diameter
- Uses BFS for distance computation
- Returns feature vector

#### 🔹 `_bfs_distances` (Lines 2702-2720)
**Location:** Lines 2702-2720  
**Technical Implementation:**
- Breadth-first search for shortest paths
- Computes distances from start node
- Returns distance dictionary

#### 🔹 `observe_other_agent` (Lines 2722-2779)
**Location:** Lines 2722-2779  
**Technical Implementation:**
- Updates Theory of Mind beliefs using particle filter
- Stores observation and action history
- Infers goal from trajectory (if ≥2 observations)
- Infers emotions from observation
- Predicts next state using velocity
- Returns belief, inferred goal, predicted state, emotions

#### 🔹 `plan_with_other_agents` (Lines 2781-2823)
**Location:** Lines 2781-2823  
**Technical Implementation:**
- Game-theoretic planning with other agents
- Computes Nash equilibrium for each agent pair
- Finds coalition of aligned agents
- Returns equilibria and coalition information

#### 🔹 `should_communicate_with` (Lines 2825-2854)
**Location:** Lines 2825-2854  
**Technical Implementation:**
- Decides if communication is valuable
- Gets belief about other agent
- Computes communication value
- Returns True if value > 0.5 threshold

#### 🔹 `simulate_multi_agent_interaction` (Lines 2856-3051)
**Location:** Lines 2856-3051  
**Technical Implementation:**
- Simulates multi-agent interaction with Theory of Mind
- Initializes agents with random states and goals
- **Simulation loop:**
  - Each agent observes others and updates beliefs
  - Game-theoretic action selection (Nash equilibrium)
  - Updates agent states based on actions
  - Coalition formation every 5 steps
  - Communication decisions based on value
- Tracks trajectories, beliefs, coalitions, communications
- Returns comprehensive simulation results


#### 🔹 `_extract_symbolic_facts` (Lines 3053-3124)
**Location:** Lines 3053-3124  
**Technical Implementation:**
- Extracts symbolic facts from neural representations
- Identifies high-activation dimensions (top 25%)
- Discretizes activation values (high/medium/low)
- Creates symbolic terms: feature(dim, polarity, concept)
- Creates concept activation facts
- Extracts relational facts (correlated/anticorrelated)
- Returns list of symbolic Term objects

#### 🔹 `learn_symbolic_rule` (Lines 3126-3148)
**Location:** Lines 3126-3148  
**Technical Implementation:**
- Learns new symbolic inference rule
- Stores rule as (premises, conclusion) tuple
- Checks for duplicates before adding
- Limits rule set to 100 rules

#### 🔹 `query_symbolic_knowledge` (Lines 3150-3164)
**Location:** Lines 3150-3164  
**Technical Implementation:**
- Queries symbolic knowledge base
- Uses resolution refutation for proof
- Returns True if query provable from KB

#### 🔹 `explain_decision_symbolically` (Lines 3166-3207)
**Location:** Lines 3166-3207  
**Technical Implementation:**
- Generates symbolic explanation for decision
- Extracts facts from current state
- Identifies high-activation features
- Checks for derived facts (implies, causes, leads_to)
- Returns list of explanation strings

#### 🔹 `_initialize_causal_hierarchy` (Lines 3209-3253)
**Location:** Lines 3209-3253  
**Technical Implementation:**
- Initializes 3-level causal abstraction hierarchy:
  - **Level 0:** Latent dimensions (fine-grained)
  - **Level 1:** Object slots (mid-level)
  - **Level 2:** Concepts (high-level)
- Sets up abstraction functions for each level
- Initializes causal graphs and intervention effects
- Creates abstraction mappings between levels

#### 🔹 `_abstract_to_objects` (Lines 3255-3263)
**Location:** Lines 3255-3263  
**Technical Implementation:**
- Abstracts latent state to object-level
- Uses object layer to segment into slots
- Extracts slot activations (mean absolute value)
- Returns object-level representation

#### 🔹 `_abstract_to_concepts` (Lines 3265-3277)
**Location:** Lines 3265-3277  
**Technical Implementation:**
- Abstracts latent state to concept-level
- Computes similarity to each concept in mesh
- Uses cosine similarity
- Returns concept activation vector (20 dimensions)

#### 🔹 `_map_latent_to_objects` (Lines 3279-3283)
**Location:** Lines 3279-3283  
**Technical Implementation:**
- Maps latent dimension indices to object slot indices
- Groups latent dims into slots
- Returns slot indices

#### 🔹 `_map_objects_to_concepts` (Lines 3285-3292)
**Location:** Lines 3285-3292  
**Technical Implementation:**
- Maps object slot indices to concept indices
- Objects can activate multiple concepts
- Returns concept indices

#### 🔹 `_map_latent_to_concepts` (Lines 3294-3296)
**Location:** Lines 3294-3296  
**Technical Implementation:**
- Direct mapping from latent to concepts
- Uses modulo operation for indexing

#### 🔹 `learn_causal_abstraction` (Lines 3298-3353)
**Location:** Lines 3298-3353  
**Technical Implementation:**
- Learns causal relationships at specified abstraction level
- Abstracts observations using level's abstraction function
- Learns causal graph using pairwise correlations
- Adds edges for strong correlations (|corr| > 0.5)
- Learns intervention effects from data
- Stores in abstraction level structure

#### 🔹 `intervene_at_abstraction` (Lines 3355-3404)
**Location:** Lines 3355-3404  
**Technical Implementation:**
- Performs intervention at specified abstraction level
- Gets current state at abstraction level
- Applies intervention (fixes variable values)
- Predicts forward using causal graph
- Updates variables based on parent influences
- Returns predicted trajectory

#### 🔹 `_build_contrastive_encoder` (Lines 3406-3417)
**Location:** Lines 3406-3417  
**Technical Implementation:**
- Builds contrastive encoder MLP
- Architecture: [256, 128, 128] hidden layers
- Projects latent to contrastive space
- Used for self-supervised learning

#### 🔹 `contrastive_learning_step` (Lines 3419-3472)
**Location:** Lines 3419-3472  
**Technical Implementation:**
- Performs contrastive learning with InfoNCE loss
- Projects anchor, positive, negatives to contrastive space
- Computes cosine similarities
- InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
- Gradient descent on contrastive encoder
- Returns loss value

#### 🔹 `generate_contrastive_pairs` (Lines 3474-3510)
**Location:** Lines 3474-3510  
**Technical Implementation:**
- Generates positive and negative pairs
- **Positive:** Recent state from buffer (temporally close)
- **Negatives:** Random distant states from buffer (8 samples)
- Adds random negatives if buffer insufficient
- Returns (positive, negatives) tuple

#### 🔹 `execute_with_options` (Lines 3512-3585)
**Location:** Lines 3512-3585  
**Technical Implementation:**
- Executes action using hierarchical options framework
- Checks if current option should terminate
- Discovers skills from trajectory if option terminates
- Selects new option if needed
- Executes option policy to get action
- Records trajectory for skill discovery
- Returns action, option info, termination status


#### 🔹 `fast_adapt` (Lines 3587-3611)
**Location:** Lines 3587-3611  
**Technical Implementation:**
- Fast adaptation using meta-learning
- Extracts gradients from support examples
- Computes prediction error for each example
- Normalizes gradients
- Calls meta_learner.adapt with 5 adaptation steps
- Updates top layer parameters with adapted values

#### 🔹 `start_new_task` (Lines 3613-3621)
**Location:** Lines 3613-3621  
**Technical Implementation:**
- Starts new task with continual learning protection
- Gets current parameters from hierarchy
- Calls continual_learner.start_new_task
- Protects important parameters from forgetting

#### 🔹 `get_statistics` (Lines 3623-3636)
**Location:** Lines 3623-3636  
**Technical Implementation:**
- Returns comprehensive statistics:
  - Step count, layer count, layer dimensions
  - Number of concepts learned
  - Memory statistics
  - Average surprise (reconstruction error)
  - Intrinsic motivation level
  - Average Expected Free Energy

#### 🔹 `visualize_hierarchy` (Lines 3638-3673)
**Location:** Lines 3638-3673  
**Technical Implementation:**
- Prints hierarchy structure and statistics
- Shows global statistics (steps, layers, concepts, motivation)
- Shows per-layer information:
  - Dimensions, reconstruction error
  - Epistemic and aleatoric uncertainty
  - Active units count
- Shows object-centric layer info
- Shows memory system statistics
- Formatted output with separators

#### 🔹 `parameters` (Lines 3675-3697)
**Location:** Lines 3675-3697  
**Technical Implementation:**
- Returns all trainable parameters from:
  - All hierarchical layers
  - Object-centric layer
  - Active inference agent
  - Hierarchical attention (if enabled)
  - Theory of Mind (if enabled)
  - Contrastive encoder
- Used for optimization and meta-learning

---

## 🔗 SECTION 3: INTEGRATION & USAGE

### 🎯 Integration with Other Modules

#### 🔄 **Integration with `nn.py`**

**Functions Used:**
- `Tensor`, `Module`, `MLP`, `Linear`, `AdaptiveNorm`

**Integration Points:**
1. **CausalPredictiveLayer.__init__** (Lines 94-148)
   - Uses `Linear` for generative model W, dynamics A_obs, action influence B
   - All neural computations use `Tensor` for automatic differentiation
   
2. **ObjectCentricPredictiveLayer.__init__** (Lines 805-843)
   - Uses `MLP` for object encoder, decoder, dynamics, relation network, physics network
   - Enables learned physics instead of hardcoded parameters

3. **AGIPredictiveHierarchy._build_contrastive_encoder** (Lines 3406-3417)
   - Uses `MLP` to build contrastive encoder for self-supervised learning

**Usage Example:**
```python
# Creating predictive layer with nn.py components
layer = CausalPredictiveLayer(input_dim=64, latent_dim=32, action_dim=4)
# W, A_obs, B are all Linear modules from nn.py
# All forward passes use Tensor for gradient tracking
```

---

#### 🧠 **Integration with `memory.py`**

**Functions Used:**
- `AGIMemorySystem`, `MemoryItem`, `WorkingMemory`, `ShortTermMemory`, `LongTermMemory`
- `VSABindingSpace`, `CognitiveRoutine`

**Integration Points:**
1. **AGIPredictiveHierarchy.__init__** (Lines 1395-1540)
   - Creates `AGIMemorySystem` with working, short-term, long-term memory
   - Initializes VSA binding space for variable binding

2. **AGIPredictiveHierarchy.process** (Lines 1542-1796)
   - **Memory Retrieval:** Queries memory with top latent (Line ~1650)
   - **Memory Encoding:** Stores experience with importance weighting (Line ~1770)
   - **Consolidation:** Periodic memory consolidation (Line ~1780)

3. **CausalPredictiveLayer.bind_variable** (Lines 768-770)
   - Uses VSA for role-filler binding in working memory

**Usage Example:**
```python
# Memory retrieval during processing
results = hierarchy.process(observation, learn=True)
retrieved_memories = results['retrieved_memories']  # From memory.py
synthesized_knowledge = results['synthesized_knowledge']  # Synthesized from memories

# Memory stores experiences with importance
# High surprise → high importance → better retention
```

---

#### 📚 **Integration with `learning_upgraded.py`**

**Functions Used:**
- `HyperbolicMeshSubstrate`, `PoincareBall`, `CausalDiscoveryEngineV2`
- `StructuralCausalModel`, `StructuralEquation`, `ContinualLearningSystem`
- `MetaLearningController`, `IntrinsicMotivationSystem`
- `HypernetworkMetaLearner`, `HierarchicalRLController`
- `BayesianUncertaintyEstimator`, `LearnedStructuralEquation`

**Integration Points:**
1. **CausalPredictiveLayer.__init__** (Lines 94-148)
   - Uses `CausalDiscoveryEngineV2` for causal structure learning
   - Integrates `BayesianUncertaintyEstimator` for uncertainty quantification

2. **CausalPredictiveLayer._update_uncertainty** (Lines 418-450)
   - Uses `BayesianUncertaintyEstimator` with 50 samples
   - Properly separates epistemic from aleatoric uncertainty

3. **CausalPredictiveLayer._discover_causal_structure** (Lines 452-497)
   - Uses PC algorithm from `CausalDiscoveryEngineV2`
   - Creates `LearnedStructuralEquation` for each variable

4. **AGIPredictiveHierarchy.__init__** (Lines 1395-1540)
   - Creates `HyperbolicMeshSubstrate` for concept learning
   - Initializes `ContinualLearningSystem` (EWC, SI, replay)
   - Sets up `MetaLearningController` (MAML, hypernetworks)
   - Creates `IntrinsicMotivationSystem` for exploration
   - Initializes `HierarchicalRLController` for options

5. **AGIPredictiveHierarchy.process** (Lines 1542-1796)
   - Updates concept mesh with hyperbolic embeddings
   - Applies continual learning updates
   - Uses intrinsic motivation for exploration

**Usage Example:**
```python
# Causal discovery in action
layer.learn(observation, action)  # Triggers causal structure learning every 50 steps
# Uses PC algorithm to discover DAG
# Creates structural equations for causal mechanisms

# Continual learning prevents forgetting
hierarchy.start_new_task(task_id=1)
# Protects important parameters with EWC
# Uses experience replay for stability
```

---

#### 👁️ **Integration with `agi_multihead_attention.py`**

**Functions Used:**
- `AGIMultiHeadSelfAttention`, `AGICausalMultiHeadAttention`

**Integration Points:**
1. **ObjectCentricPredictiveLayer.__init__** (Lines 805-843)
   - Uses `AGIMultiHeadSelfAttention` for slot attention mechanism
   - Enables competitive binding between object slots

2. **AGIPredictiveHierarchy.__init__** (Lines 1395-1540)
   - Creates hierarchical attention for cross-layer reasoning
   - Uses `AGICausalMultiHeadAttention` if available

3. **ObjectCentricPredictiveLayer.segment_objects** (Lines 845-901)
   - Applies multi-head self-attention for iterative slot refinement
   - Competitive binding: slots compete for features

**Usage Example:**
```python
# Slot attention for object segmentation
slots = object_layer.segment_objects(observation)
# Uses AGIMultiHeadSelfAttention internally
# Iterative refinement with competitive binding
```

---

#### 🌍 **Integration with `world_model.py`**

**Functions Used:**
- `WorldModel`, `PhysicsConstrainedWorldModel`, `CausalWorldModelExtension`
- `PCAlgorithmCausalDiscovery`, `ObjectPermanenceTracker`

**Integration Points:**
1. **ObjectCentricPredictiveLayer._apply_physics** (Lines 956-1021)
   - Uses `PhysicsConstrainedWorldModel` for learned physics
   - Applies constraints: permanence, causality, energy conservation

2. **ObjectCentricPredictiveLayer.track_objects** (Lines 1023-1084)
   - Uses `ObjectPermanenceTracker` with Kalman filtering
   - Handles occlusions and predicts hidden object positions

3. **CausalPredictiveLayer._discover_causal_structure** (Lines 452-497)
   - Uses `PCAlgorithmCausalDiscovery` for structure learning
   - Performs conditional independence tests

4. **AGIPredictiveHierarchy.imagine_trajectory** (Lines 1832-2025)
   - Uses `WorldModel` for Dreamer-style imagination
   - Rolls out trajectories in latent space
   - Action-conditioned predictions

**Usage Example:**
```python
# World model imagination
imagined = hierarchy.imagine_trajectory(
    initial_state=current_state,
    actions=action_sequence,
    horizon=10,
    use_world_model=True
)
# Uses WorldModel for latent rollouts
# No environment interaction needed
```

---

#### 🎯 **Integration with `active_inference_upgrades.py`**

**Functions Used:**
- `LearnedHierarchicalPlanner`, `BayesianTheoryOfMind`, `MAMLMetaLearner`
- `AttentionBasedPolicyLibrary`, `Policy`, `PolicyType`

**Integration Points:**
1. **ActiveInferencePredictiveAgent.__init__** (Lines 1125-1162)
   - Creates `LearnedHierarchicalPlanner` with 3 levels
   - Enables hierarchical planning

2. **ActiveInferencePredictiveAgent._generate_policies** (Lines 1164-1264)
   - Uses `AttentionBasedPolicyLibrary` for policy retrieval
   - NO random fallback - only learned policies
   - Composes policies for novel situations

3. **AGIPredictiveHierarchy.__init__** (Lines 1395-1540)
   - Creates `BayesianTheoryOfMind` for multi-agent reasoning
   - Enables particle filter belief tracking

4. **AGIPredictiveHierarchy.observe_other_agent** (Lines 2722-2779)
   - Uses Theory of Mind to update beliefs about other agents
   - Infers goals and emotions

**Usage Example:**
```python
# Active inference with learned policies
action = agent.select_action(goal=goal_state)
# Retrieves relevant policies from library
# Composes policies for novel situations
# Minimizes Expected Free Energy

# Theory of Mind for multi-agent
tom_result = hierarchy.observe_other_agent(
    agent_id='agent_1',
    observation=other_agent_state,
    action=other_agent_action
)
# Updates beliefs, infers goals, predicts actions
```

---

### 🚀 **Complete Usage Examples**

#### Example 1: Basic Prediction with Learning
```python
# Initialize substrate
substrate = AGIPredictiveSubstrate(
    input_dim=64,
    action_dim=4,
    layer_dims=[32, 16, 8],
    num_objects=6
)

# Process observation with learning
observation = np.random.randn(64) * 0.1
results = substrate.predict(observation, learn=True)

# Access results
latent_state = results['top_latent']
surprise = results['surprise']
retrieved_memories = results['retrieved_memories']
concept_id = results['concept_id']
```

#### Example 2: Goal-Directed Active Inference
```python
# Set goal
goal_observation = np.random.randn(64) * 0.1
substrate.set_goal(goal_observation)

# Get action that minimizes Expected Free Energy
results = substrate.predict(observation, goal=goal_observation)
action = results['action']
efe = results['efe']
intrinsic_motivation = results['intrinsic_motivation']

# Execute action in environment
next_observation = environment.step(action)
```

#### Example 3: Causal Intervention
```python
# Predict outcome of intervention
intervention = {0: 1.0, 1: -0.5}  # Set dimensions 0 and 1
trajectory = substrate.predict_intervention(
    observation,
    intervention,
    horizon=10
)

# Trajectory shows predicted states under intervention
for t, state in enumerate(trajectory):
    print(f"Step {t}: {state}")
```

#### Example 4: Counterfactual Reasoning
```python
# What would have happened if...
evidence = {0: 0.5}  # Observed value
intervention = {1: 1.0}  # Counterfactual change
query_dim = 2  # What we want to know

cf_value = substrate.counterfactual(
    observation,
    evidence,
    intervention,
    query_dim
)
print(f"Counterfactual value at dim {query_dim}: {cf_value}")
```

#### Example 5: Meta-Learning (Fast Adaptation)
```python
# Adapt to new task with few examples
task_examples = [
    {'state': np.random.randn(8), 'target': np.random.randn(8)}
    for _ in range(5)
]

substrate.fast_adapt('new_task', task_examples)
# Now adapted to new task in 5 steps!
```

#### Example 6: Multi-Agent Interaction
```python
# Observe another agent
other_agent_state = np.random.randn(8)
other_agent_action = np.array([0.5, -0.3, 0.1, 0.0])

tom_result = substrate.hierarchy.observe_other_agent(
    agent_id='agent_1',
    observation=other_agent_state,
    action=other_agent_action
)

inferred_goal = tom_result['inferred_goal']
predicted_next = tom_result['predicted_next_state']
emotions = tom_result['emotions']

# Decide if communication is valuable
should_comm = substrate.hierarchy.should_communicate_with('agent_1')
```

#### Example 7: Self-Modeling
```python
# Get self-model report
self_model = substrate.hierarchy.model_own_capabilities()

print(f"Confidence calibration: {self_model['confidence_calibration']}")
print(f"Known capabilities: {len(self_model['known_capabilities'])}")
print(f"OOD score: {self_model['out_of_distribution_score']}")
print(f"Learning progress: {self_model['learning_progress']}")
print(f"Metacognitive awareness: {self_model['metacognitive_awareness']}")
```

---

### 🔄 **Reusable Features in Other Modules**

#### Feature 1: Causal Reasoning Engine
**Where:** `CausalPredictiveLayer.predict_intervention`, `counterfactual`  
**Reuse in:**
- `world_model.py`: Causal world model predictions
- `reasoning.py`: Causal inference for planning
- `goal_driven_agi.py`: Causal goal reasoning

**How to Reuse:**
```python
from predictive_substrate import CausalPredictiveLayer

# Create causal layer
causal_layer = CausalPredictiveLayer(input_dim=64, latent_dim=32, action_dim=4)

# Use for interventional predictions
trajectory = causal_layer.predict_intervention(
    intervention={0: 1.0},
    horizon=5
)

# Use for counterfactual reasoning
cf_value = causal_layer.counterfactual(
    evidence={0: 0.5},
    intervention={1: 1.0},
    query_dim=2
)
```

#### Feature 2: Object-Centric Representations
**Where:** `ObjectCentricPredictiveLayer.segment_objects`, `predict_object_dynamics`  
**Reuse in:**
- `world_model.py`: Object-based world modeling
- `attention.py`: Object-focused attention
- `grounding.py`: Object grounding

**How to Reuse:**
```python
from predictive_substrate import ObjectCentricPredictiveLayer

# Create object layer
object_layer = ObjectCentricPredictiveLayer(
    input_dim=64,
    num_slots=6,
    slot_dim=64
)

# Segment into objects
observation = Tensor(np.random.randn(64))
slots = object_layer.segment_objects(observation)

# Predict object dynamics
action = Tensor(np.array([0.1, -0.2, 0.0, 0.3]))
next_slots = object_layer.predict_object_dynamics(action)
```

#### Feature 3: Active Inference Agent
**Where:** `ActiveInferencePredictiveAgent.select_action`, `expected_free_energy`  
**Reuse in:**
- `act.py`: Action selection
- `goal_driven_agi.py`: Goal-directed behavior
- `reasoning.py`: Planning with EFE

**How to Reuse:**
```python
from predictive_substrate import ActiveInferencePredictiveAgent, CausalPredictiveLayer

# Create predictive layer and agent
pred_layer = CausalPredictiveLayer(input_dim=64, latent_dim=32, action_dim=4)
agent = ActiveInferencePredictiveAgent(
    predictive_layer=pred_layer,
    action_dim=4,
    planning_horizon=5
)

# Set goal
goal = Tensor(np.random.randn(64))
agent.update_preferences(goal)

# Select action that minimizes EFE
action = agent.select_action(goal)
```

#### Feature 4: Uncertainty Quantification
**Where:** `CausalPredictiveLayer.get_uncertainty`, `_update_uncertainty`  
**Reuse in:**
- `learning_upgraded.py`: Uncertainty-aware learning
- `reasoning.py`: Uncertainty-aware planning
- `observe.py`: Uncertainty in perception

**How to Reuse:**
```python
# Get uncertainty decomposition
uncertainty = layer.get_uncertainty()

epistemic = uncertainty['epistemic']  # Model uncertainty (reducible)
aleatoric = uncertainty['aleatoric']  # Data noise (irreducible)
total = uncertainty['total']  # Combined

# Use for exploration
if np.mean(epistemic) > 0.5:
    # High epistemic uncertainty → explore
    action = explore_action()
else:
    # Low epistemic uncertainty → exploit
    action = exploit_action()
```

#### Feature 5: Memory Integration
**Where:** `AGIPredictiveHierarchy.process` (memory retrieval/encoding)  
**Reuse in:**
- `reasoning.py`: Memory-based reasoning
- `learning_upgraded.py`: Memory-augmented learning
- `goal_driven_agi.py`: Memory-guided goal pursuit

**How to Reuse:**
```python
# Memory retrieval during processing
results = hierarchy.process(observation, learn=True)

# Retrieved memories
memories = results['retrieved_memories']
for memory in memories:
    print(f"Memory: {memory.content}, Importance: {memory.importance}")

# Synthesized knowledge
knowledge = results['synthesized_knowledge']
```

#### Feature 6: Hierarchical Options
**Where:** `AGIPredictiveHierarchy.execute_with_options`, `discover_subgoals`  
**Reuse in:**
- `act.py`: Hierarchical action execution
- `reasoning.py`: Hierarchical planning
- `goal_driven_agi.py`: Subgoal-based goal achievement

**How to Reuse:**
```python
# Execute with options
result = hierarchy.execute_with_options(observation, goal)

action = result['action']
option_id = result['option_id']
option_terminated = result['option_terminated']
discovered_skills = result['discovered_skills']

# Discover subgoals from trajectory
trajectory = [{'state': s, 'action': a} for s, a in zip(states, actions)]
subgoals = hierarchy.discover_subgoals(trajectory)
```

#### Feature 7: Theory of Mind
**Where:** `AGIPredictiveHierarchy.observe_other_agent`, `plan_with_other_agents`  
**Reuse in:**
- `reasoning.py`: Multi-agent reasoning
- `goal_driven_agi.py`: Cooperative goal achievement
- `act.py`: Strategic action selection

**How to Reuse:**
```python
# Observe other agent
tom_result = hierarchy.observe_other_agent(
    agent_id='agent_1',
    observation=other_state,
    action=other_action
)

# Plan with other agents
plan_result = hierarchy.plan_with_other_agents(
    my_state=my_state,
    agent_states={'agent_1': other_state},
    my_actions=my_action_candidates,
    their_actions=their_action_candidates
)

nash_equilibria = plan_result['equilibria']
coalition = plan_result['coalition']
```

#### Feature 8: Self-Modeling
**Where:** `AGIPredictiveHierarchy.model_own_capabilities`  
**Reuse in:**
- `reasoning.py`: Metacognitive reasoning
- `learning_upgraded.py`: Adaptive learning based on capabilities
- `goal_driven_agi.py`: Goal selection based on capabilities

**How to Reuse:**
```python
# Get self-model
self_model = hierarchy.model_own_capabilities()

# Check if capable of task
if self_model['confidence_calibration'] > 0.7:
    # Well-calibrated, trust predictions
    proceed_with_task()
else:
    # Poorly calibrated, be cautious
    request_more_data()

# Check capability boundaries
boundaries = self_model['capability_boundaries']
for boundary in boundaries:
    print(f"Uncertain in layer {boundary['layer']}")
    print(f"Uncertain dimensions: {boundary['uncertain_dimensions']}")
```

---

## 📊 Summary

The Predictive Substrate is a comprehensive AGI-grade system implementing:
- ✅ 17 major feature categories
- ✅ 4 main classes with 80+ functions
- ✅ Deep integration with 7 core modules
- ✅ 8 reusable feature sets for other modules
- ✅ Complete causal reasoning, active inference, and metacognition

**Total Lines:** 3,951 lines of production-grade AGI code

🎉 **All features fully documented with technical implementation details!**
