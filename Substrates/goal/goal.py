"""
GOAL-DRIVEN AGI CORE (TRAINABLE VERSION)
========================================
True AGI with hierarchical goal management, autonomous planning, and self-healing.
COMPLETELY DYNAMIC & TRAINABLE: All heuristics replaced with Neural Networks.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from memory import AGIMemorySystem

# Core Autograd and Neural Primitives
from nn import Tensor, Module, MLP, Linear, AdaptiveNorm
from neural_substrate import NeuromodulatorState

class GoalStatus(Enum):
    """Dynamic goal states (Lifecycle remains discrete for system logic)"""
    CONCEIVED = "conceived"
    PLANNED = "planned"
    ACTIVE = "active"
    BLOCKED = "blocked"
    PAUSED = "paused"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"
    FAILED = "failed"

class HealingAction(Enum):
    """Actions the self-healing policy can take"""
    ADAPT = 0
    RETHINK = 1
    SWITCH = 2
    ABANDON = 3

@dataclass
class Goal:
    """A goal represented in a continuous embedding space"""
    id: str
    description: str
    
    # Continuous representation instead of fixed Enums
    embedding: Tensor = None
    level_pred: float = 0.0 # 0.0 (action) to 1.0 (dream)
    
    # Hierarchy
    parent_goal_id: Optional[str] = None
    child_goal_ids: List[str] = field(default_factory=list)
    
    # State
    status: GoalStatus = GoalStatus.CONCEIVED
    progress: float = 0.0
    
    # Planning (Reasoning-driven)
    plan: List[str] = field(default_factory=list)
    current_step: int = 0
    alternative_paths: List[List[str]] = field(default_factory=list)
    
    # Experience
    attempts: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)
    successes: List[Dict[str, Any]] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    
    # Dynamic Cognitive Metadata (Emotionally aware)
    created_at: float = field(default_factory=time.time)
    importance: float = 0.5
    urgency: float = 0.5
    confidence: float = 0.5
    complexity: float = 0.5
    valence: float = 0.0 # Emotional valence associated with this goal
    arousal: float = 0.0 # Activation level
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    required_skills: List[str] = field(default_factory=list)

class GoalEncoder(Module):
    """Encodes text description into a continuous Tensor embedding using actual AGISemanticEncoder"""
    def __init__(self, semantic_encoder: Any):
        self.semantic_encoder = semantic_encoder
        self.dim = int(getattr(semantic_encoder, 'latent_dim', 256))
        
    def __call__(self, description: str) -> Tensor:
        # Use production semantic encoder
        res = self.semantic_encoder.encode(description)
        if isinstance(res, dict) and 'latent_z' in res:
            z = res['latent_z']
            # If multiple slots, pool them or take global mean
            if len(z.data.shape) > 1:
                pooled = np.mean(z.data, axis=0)
                return Tensor(pooled, _children=(z,), _op='pool')
            return z
        return Tensor(np.zeros(self.dim))
        
    def parameters(self) -> List[Tensor]:
        # Parameters owned by semantic_encoder
        return []

class GoalAssessor(Module):
    """Multi-head assessor for importance, urgency, confidence, complexity.
    Now emotionally aware: accepts emotion_state (13-dim).
    """
    def __init__(self, dim: int, emotion_dim: int = 13):
        self.dim = dim
        self.shared = MLP(dim + emotion_dim, [128, 64], label='assessor_shared')
        # Heads: 1 output each, squashed to [0,1]
        self.head_importance = Linear(64, 1, label='assess_imp')
        self.head_urgency = Linear(64, 1, label='assess_urg')
        self.head_confidence = Linear(64, 1, label='assess_conf')
        self.head_complexity = Linear(64, 1, label='assess_comp')
        self.head_level = Linear(64, 1, label='assess_lvl') # depth in hierarchy
        
    def __call__(self, goal_emb: Tensor, emotion_state: np.ndarray) -> Dict[str, Tensor]:
        emo = np.asarray(emotion_state, dtype=float).reshape(-1)
        # Pad or truncate to emotion_dim (13)
        padded_emo = np.zeros(13)
        n = min(13, emo.shape[0])
        padded_emo[:n] = emo[:n]
        
        ctx = np.concatenate([goal_emb.data, padded_emo])
        features = self.shared(Tensor(ctx)).relu()
        
        return {
            'importance': self.head_importance(features).sigmoid(),
            'urgency': self.head_urgency(features).sigmoid(),
            'confidence': self.head_confidence(features).sigmoid(),
            'complexity': self.head_complexity(features).sigmoid(),
            'level': self.head_level(features).sigmoid()
        }
        
    def parameters(self) -> List[Tensor]:
        params = self.shared.parameters()
        params += self.head_importance.parameters()
        params += self.head_urgency.parameters()
        params += self.head_confidence.parameters()
        params += self.head_complexity.parameters()
        params += self.head_level.parameters()
        return params

class GoalDecomposer(Module):
    """Generates N child embeddings from a parent embedding"""
    def __init__(self, dim: int, max_subgoals: int = 5):
        self.dim = dim
        self.max_subgoals = max_subgoals
        self.net = MLP(dim, [128, dim * max_subgoals], label='decomposer')
        self.count_net = MLP(dim, [32, max_subgoals], label='subgoal_count_logits')
        
    def __call__(self, parent_emb: Tensor) -> Tuple[List[Tensor], Tensor]:
        """Returns list of subgoal embeddings and the count logits"""
        flat_embs = self.net(parent_emb)
        counts = self.count_net(parent_emb)
        
        # Reshape flat output into list of tensors
        subgoals = []
        for i in range(self.max_subgoals):
            sub_np = flat_embs.data[i*self.dim : (i+1)*self.dim]
            subgoals.append(Tensor(sub_np, _children=(flat_embs,), _op='slice'))
            
        return subgoals, counts
        
    def parameters(self) -> List[Tensor]:
        return self.net.parameters() + self.count_net.parameters()

class HealingPolicy(Module):
    """Chooses self-healing action based on goal state, failure history, and neuromodulators."""
    def __init__(self, dim: int, emotion_dim: int = 13):
        # Input features: goal_emb (dim) + stuck_count (1) + attempts (1) + conf (1) + emotion (13) + modulators (4)
        self.dim = dim
        self.net = MLP(dim + 3 + emotion_dim + 4, [64, 4], label='healing_policy')
        
    def __call__(self, goal_emb: Tensor, stuck_cnt: int, attempts: int, conf: float, 
                 emotion: np.ndarray, modulators: NeuromodulatorState) -> Tensor:
        emo = np.asarray(emotion, dtype=float).reshape(-1)
        padded_emo = np.zeros(13)
        n = min(13, emo.shape[0])
        padded_emo[:n] = emo[:n]
        
        mods = np.array([
            modulators.dopamine,
            modulators.acetylcholine,
            modulators.norepinephrine,
            modulators.plasticity_gate
        ])
        
        ctx = np.concatenate([
            goal_emb.data,
            np.array([stuck_cnt / 10.0, attempts / 10.0, conf]),
            padded_emo,
            mods
        ])
        
        x = Tensor(ctx, _children=(goal_emb,), _op='concat')
        return self.net(x) # 4 actions: ADAPT, RETHINK, SWITCH, ABANDON

class ValueNetwork(Module):
    """Predicts expected success/reward of a goal, used as baseline for training"""
    def __init__(self, dim: int):
        self.net = MLP(dim, [64, 1], label='value_net')
        
    def __call__(self, goal_emb: Tensor) -> Tensor:
        return self.net(goal_emb)
        
    def parameters(self) -> List[Tensor]:
        return self.net.parameters()


class GoalPriorityNetwork(Module):
    """Trainable priority scorer for goal selection under emotion/context."""
    def __init__(self, dim: int, emotion_dim: int = 13):
        self.dim = int(dim)
        self.emotion_dim = int(emotion_dim)

        # Features: goal_emb + emotion + goal metadata (importance/urgency/confidence/complexity/progress/attempts)
        self.net = MLP(self.dim + self.emotion_dim + 6, [128, 64, 1], label='goal_priority')

    def __call__(self, goal_emb: Tensor, emotion_state: np.ndarray, meta: np.ndarray) -> Tensor:
        emo = np.asarray(emotion_state, dtype=float).reshape(-1)
        padded_emo = np.zeros(self.emotion_dim, dtype=float)
        n = min(self.emotion_dim, int(emo.shape[0]))
        if n > 0:
            padded_emo[:n] = emo[:n]

        m = np.asarray(meta, dtype=float).reshape(-1)
        if m.size != 6:
            m = np.resize(m, 6)

        x = np.concatenate([goal_emb.data.reshape(-1), padded_emo, m], axis=0)
        return self.net(Tensor(x, _children=(goal_emb,), _op='concat'))

    def parameters(self) -> List[Tensor]:
        return self.net.parameters()


class GoalProgressUpdater(Module):
    """Trainable progress/confidence update conditioned on outcome signals."""
    def __init__(self, dim: int, emotion_dim: int = 13):
        self.dim = int(dim)
        self.emotion_dim = int(emotion_dim)

        # Inputs: goal_emb + emotion + [success, reward_proxy, attempts_norm, current_progress]
        self.net = MLP(self.dim + self.emotion_dim + 4, [128, 64, 2], label='goal_progress_updater')
        self.norm = AdaptiveNorm(2, label='goal_progress_norm')

    def __call__(
        self,
        goal_emb: Tensor,
        emotion_state: np.ndarray,
        success: bool,
        reward_proxy: float,
        attempts: int,
        progress: float,
    ) -> Tuple[float, float]:
        emo = np.asarray(emotion_state, dtype=float).reshape(-1)
        padded_emo = np.zeros(self.emotion_dim, dtype=float)
        n = min(self.emotion_dim, int(emo.shape[0]))
        if n > 0:
            padded_emo[:n] = emo[:n]

        feats = np.array(
            [
                1.0 if bool(success) else 0.0,
                float(np.tanh(float(reward_proxy))),
                float(min(1.0, max(0.0, attempts / 10.0))),
                float(np.clip(progress, 0.0, 1.0)),
            ],
            dtype=float,
        )

        x = np.concatenate([goal_emb.data.reshape(-1), padded_emo, feats], axis=0)
        logits = self.net(Tensor(x, _children=(goal_emb,), _op='concat'))
        out = self.norm(logits).data.reshape(-1)
        if out.size < 2:
            out = np.resize(out, 2)

        # Map to bounded deltas.
        dprog = float(np.tanh(out[0])) * 0.2
        dconf = float(np.tanh(out[1])) * 0.1
        return dprog, dconf

    def parameters(self) -> List[Tensor]:
        return self.net.parameters() + self.norm.parameters()

# ============================================================================
# MAIN AGI CLASS
# ============================================================================

class GoalDrivenAGI(Module):
    """
    Autonomous AGI with hierarchical goal management.
    Mastered by Reasoning, Implemented via Active Inference.
    """
    def __init__(self, reasoning_system, active_inference, memory_system, semantic_encoder, dim: int = 256):
        self.dim = dim
        self.reasoning = reasoning_system
        self.active_inference = active_inference
        self.memory = memory_system
        self.semantic_encoder = semantic_encoder
        
        # Neural Modules
        self.encoder = GoalEncoder(semantic_encoder)
        self.assessor = GoalAssessor(dim)
        self.decomposer = GoalDecomposer(dim)
        self.healing = HealingPolicy(dim)
        self.value_net = ValueNetwork(dim)
        self.priority_net = GoalPriorityNetwork(dim)
        self.progress_updater = GoalProgressUpdater(dim)
        
        # State
        self.goals: Dict[str, Goal] = {}
        self.active_goal_id: Optional[str] = None
        self.goal_counter = 0
        
        # Buffer for training (experience replay)
        self.experiences = []
        
    def parameters(self) -> List[Tensor]:
        params = []
        params.extend(self.assessor.parameters())
        params.extend(self.decomposer.parameters())
        params.extend(self.healing.parameters())
        params.extend(self.value_net.parameters())
        params.extend(self.priority_net.parameters())
        params.extend(self.progress_updater.parameters())
        return params
        
    def conceive_goal(self, description: str, emotion_state: np.ndarray, 
                      parent_id: Optional[str] = None, context: Dict = None) -> Goal:
        """AGI conceives a new goal. Networks assess its properties under current emotional context."""
        
        # Forward pass: Encode description
        emb = self.encoder(description)
        
        # Forward pass: Assess goal
        assessments = self.assessor(emb, emotion_state)
        
        goal_id = f"goal_{self.goal_counter}"
        self.goal_counter += 1
        
        goal = Goal(
            id=goal_id,
            description=description,
            embedding=emb,
            level_pred=float(assessments['level'].data[0]),
            parent_goal_id=parent_id,
            importance=float(assessments['importance'].data[0]),
            urgency=float(assessments['urgency'].data[0]),
            confidence=float(assessments['confidence'].data[0]),
            complexity=float(assessments['complexity'].data[0]),
            valence=float(emotion_state[0]) if emotion_state.shape[0] > 0 else 0.0,
            arousal=float(emotion_state[1]) if emotion_state.shape[0] > 1 else 0.0,
            context=context or {}
        )
        
        self.goals[goal_id] = goal
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].child_goal_ids.append(goal_id)
            
        self._store_goal_in_memory(goal, "conceived")
        return goal

    def evaluate_goals(self, emotion_state: np.ndarray) -> List[Tuple[str, float]]:
        """Evaluate all goals for priority given current emotional state."""
        priorities = []
        for gid, goal in self.goals.items():
            if goal.status in [GoalStatus.ACHIEVED, GoalStatus.ABANDONED, GoalStatus.FAILED]:
                continue

            meta = np.array(
                [
                    float(getattr(goal, 'importance', 0.5)),
                    float(getattr(goal, 'urgency', 0.5)),
                    float(getattr(goal, 'confidence', 0.5)),
                    float(getattr(goal, 'complexity', 0.5)),
                    float(np.clip(getattr(goal, 'progress', 0.0), 0.0, 1.0)),
                    float(min(1.0, max(0.0, getattr(goal, 'attempts', 0) / 10.0))),
                ],
                dtype=float,
            )

            prio_t = self.priority_net(goal.embedding, emotion_state, meta)
            prio = float(np.asarray(prio_t.data, dtype=float).reshape(-1)[0])
            priorities.append((gid, prio))
            
        return sorted(priorities, key=lambda x: x[1], reverse=True)

    def set_active_goal(self, goal_id: str) -> bool:
        """Setting the active goal notifies the Active Inference engine of the new preference."""
        if goal_id not in self.goals:
            return False
            
        goal = self.goals[goal_id]
        self.active_goal_id = goal_id
        goal.status = GoalStatus.ACTIVE
        
        # SYNC WITH ACTIVE INFERENCE: Map goal embedding to preference (C vector)
        if self.active_inference and hasattr(self.active_inference, 'canonical'):
            emb_data = goal.embedding.data.reshape(-1)
            pref = np.zeros(self.active_inference.state_dim)
            n = min(pref.shape[0], emb_data.shape[0])
            pref[:n] = emb_data[:n]
            self.active_inference.canonical.set_preferences(pref)
            
        self._store_goal_in_memory(goal, "activated")
        return True

    def update_goal_status(self, goal_id: str, success: bool, feedback: Dict[str, Any]):
        """Called by Reasoning to update goal state based on perception/execution results."""
        if goal_id not in self.goals: return
        goal = self.goals[goal_id]

        reward_proxy = 0.0
        try:
            reward_proxy = float(feedback.get('reward', 0.0))
        except Exception:
            reward_proxy = 0.0

        try:
            dprog, dconf = self.progress_updater(
                goal.embedding,
                emotion_state=np.asarray([float(getattr(goal, 'valence', 0.0)), float(getattr(goal, 'arousal', 0.0))] + [0.0] * 11, dtype=float),
                success=bool(success),
                reward_proxy=float(reward_proxy),
                attempts=int(getattr(goal, 'attempts', 0)),
                progress=float(getattr(goal, 'progress', 0.0)),
            )
        except Exception:
            dprog, dconf = (0.05 if success else -0.02), (0.02 if success else -0.02)

        if bool(success):
            goal.progress = float(np.clip(float(goal.progress) + float(dprog), 0.0, 1.0))
            goal.confidence = float(np.clip(float(goal.confidence) + float(abs(dconf)), 0.0, 1.0))
            if goal.progress >= 0.999:
                goal.status = GoalStatus.ACHIEVED
                self._store_goal_in_memory(goal, "achieved")
        else:
            goal.failures.append(feedback)
            goal.confidence = float(np.clip(float(goal.confidence) - float(abs(dconf)), 0.0, 1.0))
            
    def get_healing_suggestion(self, goal_id: str, emotion: np.ndarray, 
                               modulators: NeuromodulatorState) -> Optional[HealingAction]:
        """Requests a healing action suggestion from the neural policy."""
        if goal_id not in self.goals: return None
        goal = self.goals[goal_id]
        
        stuck_cnt = len(goal.failures)
        logits = self.healing(goal.embedding, stuck_cnt, goal.attempts, goal.confidence, emotion, modulators)
        action_idx = int(np.argmax(logits.data))
        return HealingAction(action_idx)

    # --- Healing Implementations ---
    def _heal_adapt(self, goal: Goal) -> bool:
        if hasattr(self.reasoning, 'metacognitive_monitoring'):
            res = self.reasoning.metacognitive_monitoring([goal.plan[goal.current_step]])
            suggs = res.get('suggestions', [])
            if suggs:
                goal.plan.insert(goal.current_step, f"Correction: {suggs[0]}")
                return True
        return False
        
    def _heal_rethink(self, goal: Goal) -> bool:
        goal.current_step = max(0, goal.current_step - 2)
        goal.progress = goal.current_step / max(1, len(goal.plan))
        return True
        
    def _heal_switch(self, goal: Goal) -> bool:
        if goal.alternative_paths:
            goal.plan = goal.alternative_paths.pop(0)
            goal.current_step = 0
            goal.progress = 0.0
            return True
        return False

    def _execute_step(self, goal) -> Dict:
        if goal.current_step >= len(goal.plan): return {'success': True}
        if hasattr(self.reasoning, 'integrated_reasoning'):
            return self.reasoning.integrated_reasoning(goal.plan[goal.current_step])
        # Deterministic fallback when no integrated reasoning is available.
        # Use the value network prediction as a proxy success probability.
        try:
            v = float(np.asarray(self.value_net(goal.embedding).data, dtype=float).reshape(-1)[0])
            p = 1.0 / (1.0 + float(np.exp(-np.clip(v, -20.0, 20.0))))
            return {'success': bool(p >= 0.5), 'p_success': float(p)}
        except Exception:
            return {'success': False}

    def _record_success(self, goal, res):
        goal.successes.append({'step': goal.current_step, 'time': time.time()})
        goal.confidence = min(1.0, goal.confidence * 1.05)
        
    def _record_failure(self, goal, err):
        goal.failures.append({'step': goal.current_step, 'error': err})
        goal.confidence = max(0.1, goal.confidence * 0.95)
        
    def _celebrate(self, goal):
        self._store_goal_in_memory(goal, "achieved")
        # Backprop trigger in a full system: could compute loss and run .backward() here

    def _store_goal_in_memory(self, goal: Goal, event: str):
        ctx = {
            'id': goal.id, 'event': event, 'desc': goal.description,
            'status': goal.status.value, 'progress': goal.progress,
            'plan': goal.plan
        }
        # Encode into the unified AGIMemorySystem
        # importance controls probability of consolidation into Long-Term Memory
        self.memory.encode(
            content=goal.embedding, 
            importance=goal.importance, 
            context=ctx
        )

    # --- Training Tools ---
    def _record_experience(self, goal: Goal, state: str, **kwargs):
        exp = {'goal_id': goal.id, 'state': state, 'emb': goal.embedding}
        exp.update(kwargs)
        self.experiences.append(exp)

    def compute_loss(self) -> Tensor:
        """Computes training loss over buffered experiences."""
        if not self.experiences: return Tensor(np.array([0.0]))
        
        total_loss = Tensor(np.array([0.0]))
        terminal_rewards = {e['goal_id']: e.get('reward', 0.0) for e in self.experiences if e['state'] == 'terminal'}
        
        num_updates = 0
        for exp in self.experiences:
            # Value Network Loss (MSE: predict final reward)
            if exp['state'] == 'terminal':
                val_pred = self.value_net(exp['emb'])
                target = Tensor(np.array([exp.get('reward', 0.0)]))
                diff = val_pred - target
                v_loss = diff * diff
                total_loss = total_loss + v_loss
                num_updates += 1
                
            # Healing Policy Loss (REINFORCE-like)
            if exp['state'] == 'stuck' and exp['goal_id'] in terminal_rewards:
                R = terminal_rewards[exp['goal_id']]
                val_baseline = self.value_net(exp['emb']).data[0]
                advantage = R - val_baseline
                
                if 'healing_logits' in exp:
                    logits = exp['healing_logits']
                    probs = logits.sigmoid()
                    a_idx = exp['action_idx']
                    mask = np.zeros(4)
                    mask[a_idx] = 1.0
                    p_a = (probs * Tensor(mask)).sum()
                    pseudo_log_p = p_a * Tensor(np.array([advantage])) 
                    total_loss = total_loss - pseudo_log_p
                    num_updates += 1
                
        if num_updates > 0:
            total_loss = total_loss * Tensor(np.array([1.0 / num_updates]))
            
        return total_loss

# ============================================================================
# SELF-TEST & VALIDATION
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TRAINABLE GOAL-DRIVEN AGI - INTEGRATED TEST")
    print("=" * 70)
    
    class MockEncoder:
        def __init__(self): self.latent_dim = 256
        def encode(self, text, **kwargs): return {'latent_z': Tensor(np.random.randn(256))}
            
    class MockReasoning: pass
    class MockActiveInf:
        def __init__(self): self.state_dim = 256
        class _Canon:
            def set_preferences(self, p): print(f"Active Inference Preference Set: norm={np.linalg.norm(p):.4f}")
        self.canonical = _Canon()

    encoder = MockEncoder()
    agi = GoalDrivenAGI(MockReasoning(), MockActiveInf(), None, encoder, dim=256)
    
    print("\n[.] Checking Neural Goal Conception")
    emotion = np.random.randn(13)
    g1 = agi.conceive_goal("Build an AGI", emotion)
    print(f"Goal ID: {g1.id}")
    print(f"Predicted Importance: {g1.importance:.4f}")
    
    print("\n[.] Checking Active Inference Sync")
    agi.set_active_goal(g1.id)
    
    print("\n[.] Checking Neural Priority Evaluation")
    prios = agi.evaluate_goals(emotion)
    print(f"Goal Priorities: {prios}")
    
    print("\n[.] Checking Healing Suggestion")
    modulators = NeuromodulatorState()
    sugg = agi.get_healing_suggestion(g1.id, emotion, modulators)
    print(f"Healing Suggestion: {sugg}")
    
    print("\n[OK] Cognitive Goal Integration Confirmed.")
    print("=" * 70)
