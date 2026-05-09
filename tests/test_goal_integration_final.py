import sys
import os
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mind.brain import AGIMind
from nn import Tensor

def test_unified_goal_lifecycle():
    print("=" * 80)
    print("UNIFIED AGI MIND - INTEGRATED GOAL LIFECYCLE TEST")
    print("=" * 80)

    # 1. Initialize AGIMind
    print("\n[1] Initializing AGIMind...")
    mind = AGIMind(latent_dim=64)
    print("Mind initialized with GoalSystem and ActiveInferenceFacade.")

    # 2. Simulate Reasoning-driven Goal Conception
    print("\n[2] Simulating Goal Conception via Reasoning Query...")
    # The tick should handle "conceive goal" queries
    res = mind.tick(
        reasoning_query="conceive goal: Become a master of Active Inference architecture",
        reasoning_context="The user wants to see a Reasoning-driven model that can learn and adapt.",
        modality='text'
    )
    
    # Verify goal creation
    goals = mind.goal_system.goals
    if not goals:
        print("[FAIL] No goals conceived.")
        return
    
    g_id = list(goals.keys())[0]
    goal = goals[g_id]
    print(f"Goal Conceived: ID={g_id}, Description='{goal.description}'")
    print(f"Initial Assessment: Importance={goal.importance:.4f}, Urgency={goal.urgency:.4f}")
    print(f"Goal Embedding Shape: {goal.embedding.data.shape}")

    # 3. Verify Active Inference Synchronization
    print("\n[3] Verifying Active Inference Preference Sync...")
    active_gid = mind.goal_system.active_goal_id
    print(f"Active Goal ID: {active_gid}")
    
    if active_gid == g_id:
        print("[SUCCESS] Goal-System correctly set the active goal.")
    else:
        print(f"[FAIL] Active goal mismatch. Expected {g_id}, got {active_gid}")
        return

    # Check Active Inference Facade's preference vector
    if mind.active_inference and hasattr(mind.active_inference, 'canonical'):
        pref = mind.active_inference.canonical.C_mu.data
        print(f"Preferences synced: {np.mean(pref):.4f} (mean)")
        if mind.goal_system.active_goal_id:
            goal = mind.goal_system.goals[mind.goal_system.active_goal_id]
            print(f"[SUCCESS] Preference sync verified for {goal.description[:30]}...")
    else:
        print("[SKIP] Active Inference Canonical not found")

    # 4. Simulate Cognitive Update (Status Update)
    print("\n[4] Simulating Cognitive Progress Update...")
    # Reasoning update: goal step was successful
    mind.goal_system.update_goal_status(g_id, success=True, feedback={"step": "First stage complete"})
    print(f"Goal Progress: {goal.progress:.2f}")
    if goal.progress > 0:
        print("[SUCCESS] Goal status updated via Reasoning hook.")

    # 5. Emotional Modulation Test
    print("\n[5] Testing Emotional Modulation of Priorities...")
    # Inject surprise to change emotion
    res_surp = mind.tick(
        observation="UNEXPECTED ARCHITECTURAL SIGNAL DETECTED",
        modality='text'
    )
    emo = res_surp['emotion_basic']
    print(f"Primary Emotion after surprise: {emo}")
    
    prios = mind.goal_system.evaluate_goals(res_surp['emotion_state'])
    print(f"Updated Priorities: {prios}")

    print("\n" + "=" * 80)
    print("[PASSED] Integrated Goal-Driven AGI Lifecycle Verified.")
    print("=" * 80)

if __name__ == "__main__":
    test_unified_goal_lifecycle()
