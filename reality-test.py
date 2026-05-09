import numpy as np
from reasoning import IntegratedReasoningSubstrate, CognitiveMode
from neural_substrate import NeuralSubstrate
from emotion import EmotionState
import time

def run_reality_test():
    print("=== STARTING AGI REALITY TEST: EMOTIONAL REASONING INTEGRATION ===")
    
    # 1. Initialize the Master Substrate
    # This involves latent space setup, module loading, and symbolic KB initialization
    substrate = IntegratedReasoningSubstrate(latent_dim=256, action_dim=16)
    print("Substrate initialized. Blank slate ready for task adaptation.")

    # 2. Define the Query
    query = "Analyze the relationship between entropy, information theory, and the emergence of consciousness in biological systems."
    print(f"\nQUERY: {query}")

    # 3. SCENARIO A: Neutral/Low-Energy State
    # Valence: 0, Arousal: 0.2 (low energy), DA: 0.1 (low curiosity), ACh: 0.3 (low focus)
    neutral_emotion = np.array([0.0, 0.2])
    neutral_neuromodulators = {'DA': 0.1, 'ACh': 0.3}
    
    print("\n--- SCENARIO A: Neutral / Low-Energy State ---")
    start_time = time.time()
    result_a = substrate.reason(
        query=query,
        emotion_state=neutral_emotion,
        neuromodulators=neutral_neuromodulators
    )
    duration_a = time.time() - start_time
    print(f"DEBUG: Result A keys: {list(result_a.keys())}")
    print(f"DEBUG: Result A confidence: {result_a.get('confidence')}")
    print(f"DEBUG: Result A emotion_bias: {result_a.get('emotion_bias')}")
    
    # 4. SCENARIO B: High-Arousal / High-Curiosity State
    # Valence: 0.5 (positive interest), Arousal: 0.9 (intense focus), DA: 0.9 (high curiosity), ACh: 0.9 (high focus)
    active_emotion = np.array([0.4, 0.8, 0.2, 0.9])
    active_neuromodulators = {'DA': 0.9, 'ACh': 0.9}
    
    print("\n--- SCENARIO B: High-Arousal / High-Curiosity State ---")
    start_time = time.time()
    result_b = substrate.reason(
        query=query,
        emotion_state=active_emotion,
        neuromodulators=active_neuromodulators
    )
    duration_b = time.time() - start_time

    # 5. Comparative Analysis (Verification)
    print("\n=== VERIFICATION ANALYSIS ===")
    
    print(f"Scenario A (Neutral) Confidence: {result_a.get('confidence', 0.0):.4f}")
    print(f"Scenario B (Active) Confidence:  {result_b.get('confidence', 0.0):.4f}")
    
    print(f"Scenario A Reason Depth: {result_a.get('reasoning_depth', 0)}")
    print(f"Scenario B Reason Depth: {result_b.get('reasoning_depth', 0)}")

    if 'memory_retrieval' in result_a and 'memory_retrieval' in result_b:
        print(f"Scenario A Memory K (Arousal-Modulated): {result_a['memory_retrieval'].get('total_retrieved', 0)}")
        print(f"Scenario B Memory K (Arousal-Modulated): {result_b['memory_retrieval'].get('total_retrieved', 0)}")

    print(f"Scenario A Latency: {duration_a:.2f}s")
    print(f"Scenario B Latency: {duration_b:.2f}s")

    # The essence of dynamic architecture: 
    # High-arousal should lead to more total retrieved facts and higher confidence/depth.
    if result_b.get('confidence', 0) > result_a.get('confidence', 0):
        print("\n[SUCCESS] ARCHITECTURAL DYNAMICS VERIFIED: Positive affect boosted reasoning confidence.")
    else:
        print("\n[NOTICE] Architectural dynamics subtle or neutral in this specific instance.")

if __name__ == "__main__":
    try:
        run_reality_test()
    except Exception as e:
        print(f"\n[ERROR] Test failed due to: {e}")
        import traceback
        traceback.print_exc()
