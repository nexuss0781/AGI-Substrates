# Quarantine Log: Attention Substrate
*Status: Deferred for Future Training*

This document isolates the powerful, trainable AGI-grade components from `attention.py` that were intentionally deferred from the Phase 1 Integration (Master Mind). 

Our current strategy prioritizes components strictly necessary for the AGI to understand and follow English instructions. The below functions are completely AGI-grade but will introduce immense training bottlenecks if included in Phase 1. They are quarantined here, along with a high-level, non-technical overview of how they will be trained in the future.

---

## 1. Theory of Mind Module (`TheoryOfMindModule` / `MultiAgentAttentionCoordinator`)
**Objective**: To allow the AGI to understand that other entities (like the human user or other AI agents) have their own independent thoughts, beliefs, and intents that may differ from its own.

**Features**:
*   *Belief Recognition*: Guesses what another person is thinking based on their actions.
*   *Deception Detection*: Flags when an entity's actions don't match their stated intent.
*   *Belief Divergence*: Measures how far off the AGI's worldview is from the human's worldview.

**How to Train (Future)**:
*   **Dataset Needed**: Videos, text logs, or interactive game scenarios involving multi-agent cooperation, negotiation, and deception (e.g., hidden role games like "Among Us" or "Diplomacy").
*   **Training Strategy**: 
    1.  Show the AGI a scenario where Agent A hides an item from Agent B.
    2.  Ask the AGI: "Where does Agent B *think* the item is?"
    3.  Reward the AGI for correctly simulating Agent B's false belief.

---

## 2. Causal Discovery Learner (`CausalGraphLearner` / `CounterfactualReasoner`)
**Objective**: To empower the AGI to understand true "cause and effect" rather than just statistical correlations, allowing it to ask "What if I did things differently?"

**Features**:
*   *Interventional Learning*: The AGI actively pokes the world (intervenes) to see what changes, mapping out a direct causal graph.
*   *Counterfactual Reasoning*: If an action fails, it replays the memory and calculates "If I had paid attention to X instead of Y, would I have succeeded?"

**How to Train (Future)**:
*   **Dataset Needed**: Interactive sandbox environments (like a physics engine or a simulated chemistry lab) where variables can be directly manipulated.
*   **Training Strategy**:
    1.  Place the AGI in an environment where flipping Switch A turns on Light B, but Switch C does nothing.
    2.  Let the AGI randomly intervene (press switches).
    3.  Reward the AGI for correctly generating a causal map proving `A -> B`.
    4.  Test it by giving it a broken light and asking "What caused this?"

---

## 3. Predictive World Model & Planner (`PredictiveWorldModel` / `SteinPlanner`)
**Objective**: To allow the AGI to mentally simulate the future and plan complex, multi-step strategies before taking a single physical action.

**Features**:
*   *Latent Imagination*: Rolls out possible future states in its "mind's eye" without needing to physically act.
*   *Diverse Path Planning*: Generates multiple, totally different strategies to achieve the same goal (Amortized Stein Planning).

**How to Train (Future)**:
*   **Dataset Needed**: Long-horizon video data (e.g., full videos of someone assembling furniture or cooking a meal) or long-form strategic games (e.g., Chess, Go, StarCraft).
*   **Training Strategy**:
    1.  Show the AGI the first 5 seconds of a video or game state.
    2.  Ask the AGI to output what the state will look like 10, 20, and 60 seconds into the future.
    3.  Reward the AGI for accurate imagination (low prediction error).
    4.  Give the AGI a goal and reward it for mentally finding a path to that goal that avoids obstacles.

---

## 4. Object-Based Attention & Tracking (`ObjectBasedAttention` / `ObjectTracker`)
**Objective**: To give the AGI a deep, human-like understanding of physical objects—knowing that an object still exists even when hidden, and knowing what can be done with it.

**Features**:
*   *Object Permanence*: Remembers where an object is even if it gets covered up or goes off-screen.
*   *Affordance Learning*: Learns what an object is "for" (e.g., a cup is for drinking, a button is for pressing).

**How to Train (Future)**:
*   **Dataset Needed**: High-speed video tracking datasets, autonomous driving datasets, and robotics manipulation datasets (e.g., videos of humans interacting with varied objects).
*   **Training Strategy**:
    1.  Show the AGI a video of a ball rolling behind a wall.
    2.  Reward the AGI for keeping an active "attention pointer" locked on the invisible ball behind the wall.
    3.  Show the AGI varied objects and reward it for correctly predicting the object's affordance (e.g., "graspable", "pushable").

---

## 5. Curiosity & Metacognition (`CuriosityModule` / `MetacognitiveMonitor`)
**Objective**: To make the AGI self-driven and self-aware of its own limitations, prompting it to learn independently and ask for help when it is confused.

**Features**:
*   *Intrinsic Motivation (Curiosity)*: Actively seeks out novel or surprising situations to learn from them.
*   *Confidence Estimation*: Calculates how confident it is in its current action or answer.
*   *Strategy Revision*: Realizes when its current approach is failing and dynamically switches tactics.

**How to Train (Future)**:
*   **Dataset Needed**: Open-world exploration environments (e.g., Minecraft) and datasets containing human self-correction (e.g., math problem solving where the student catches their own mistake).
*   **Training Strategy**:
    1.  Place the AGI in a completely unknown, unrewarded environment.
    2.  Provide a permanent internal reward signal simply for finding something new or unpredictable.
    3.  To train metacognition, give the AGI intentionally impossible tasks. Reward it not for succeeding, but for accurately reporting "I have low confidence and need to revise my strategy."

---

## 6. Global Workspace / Consciousness Substrate (`ConsciousnessSubstrate`)
**Objective**: To create a centralized "conscious thought" arena where the AGI's most important, highly-activated thoughts are broadcast to all other sub-modules simultaneously.

**Features**:
*   *Integrated Information Computer*: Measures the mathematical "consciousness" level (Phi, Φ) of the current thought.
*   *Global Broadcasting*: Takes a thought from a single module (e.g., "I see fire") and instantly shares it with memory, action, and reasoning modules.

**How to Train (Future)**:
*   **Dataset Needed**: Extremely complex, multi-modal tasks requiring coordination across vision, text, logic, and physical action simultaneously (e.g., real-world robotics acting on dynamic verbal instructions).
*   **Training Strategy**:
    1.  This is less about a specific dataset and more about architectural routing.
    2.  Reward the system for successfully solving tasks that require disparate modules (e.g., visual perception + acoustic parsing) to share a single "eureka" concept.
    3.  Train a competition network where modules bid for priority space in the conscious workspace.

---

## 7. Meta-Attention Controller (`MetaAttentionController`)
**Objective**: To allow the AGI to dynamically govern *how* it pays attention, acting like a skilled conductor managing an orchestra.

**Features**:
*   *Strategy Selection*: Dynamically toggles between 8 distinct attention styles (e.g., Bottom-Up, Top-Down, Exploratory, Symbolic) based on what the current task demands.

**How to Train (Future)**:
*   **Dataset Needed**: A highly randomized curriculum of tasks ranging in style—some requiring fast reaction (bottom-up), some requiring deep logic (symbolic), some requiring finding a needle in a haystack (exploratory).
*   **Training Strategy**:
    1.  Feed the AGI tasks from the randomized curriculum without explicitly telling it what kind of task it is.
    2.  Reward the AGI heavily when the Meta-Attention Controller successfully predicts and locks in the optimal strategy for the hidden task type.
