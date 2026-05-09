import math
import unittest
import warnings

import numpy as np

import active_inference_upgrades as m


def _seed_all(seed: int = 0) -> None:
    np.random.seed(seed)


class TestAGIGradeActiveInferenceUpgrades(unittest.TestCase):
    def setUp(self):
        _seed_all(0)

    def test_efe_compute_is_finite_and_components_consistent(self):
        calc = m.AGIGradeEFECalculator(state_dim=8, action_dim=3, num_hypotheses=3)

        ctx = np.random.randn(8)
        w = calc.train_step(
            ctx,
            {
                "pragmatic": 1.0,
                "epistemic": 0.0,
                "novelty": 0.0,
                "empowerment": 0.0,
                "social": 0.0,
            },
        )
        self.assertEqual(np.asarray(w).shape[0], 5)

        states = [np.random.randn(8) for _ in range(5)]
        unc = [np.ones(8) * 0.1 for _ in range(5)]
        acts = [np.random.randn(3) for _ in range(5)]

        total, comp = calc.compute_efe(
            states,
            unc,
            acts,
            goal=np.zeros(8),
            other_agents=[{"state": np.ones(8), "weight": 1.0}, {"state": -np.ones(8), "weight": 0.5}],
        )
        self.assertTrue(np.isfinite(total))
        self.assertTrue(all(k in comp for k in ("pragmatic", "epistemic", "novelty", "empowerment", "social", "risk", "action_cost")))
        self.assertTrue(all(np.isfinite(v) for v in comp.values()))

    def test_td_update_and_batch_train_step_runs(self):
        td = m.TDCreditAssignment(state_dim=8, action_dim=3)

        for _ in range(50):
            s = np.random.randn(8)
            ns = np.random.randn(8)
            a = np.random.randn(3)
            r = float(np.random.randn())
            de = td.compute_td_error(s, a, r, ns, False)
            td.update_value_function(s, de, learning_rate=0.01)

        loss = td.train_step(batch_size=16, learning_rate=0.001)
        self.assertTrue(loss is None or (isinstance(loss, float) and np.isfinite(loss)))

    def test_symbolic_dynamic_vocab_safe_full(self):
        si = m.AGISymbolicInterface(state_dim=8, initial_vocab_size=10, max_vocab_size=12)
        initial = si.get_vocabulary_size()

        for i in range(50):
            si.add_word(f"w{i}")

        self.assertLessEqual(si.get_vocabulary_size(), 12)
        self.assertGreaterEqual(initial, 1)

    def test_symbolic_parse_and_compose_runnable(self):
        si = m.AGISymbolicInterface(state_dim=16, initial_vocab_size=50, max_vocab_size=200)
        meaning = si.encode_utterance("the red cube is on the table")
        self.assertEqual(meaning.data.shape[0], 16)
        utt = si.decode_to_utterance(meaning, max_length=5)
        self.assertIsInstance(utt, str)

    def test_dynamics_learn_and_train_step(self):
        planner = m.LearnedDynamicsPlanner(state_dim=8, action_dim=3, num_models=2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error", RuntimeWarning)

            for _ in range(80):
                s = np.random.randn(8)
                a = np.random.randn(3)
                ns = s + 0.01 * np.pad(a, (0, 5))
                planner.learn_dynamics(s, a, ns)

            ok = planner.train_step(batch_size=16, learning_rate=0.01)
            self.assertIsInstance(ok, bool)
            self.assertEqual(len(w), 0)

    def test_mcts_plan_returns_actions(self):
        planner = m.LearnedDynamicsPlanner(state_dim=8, action_dim=3, num_models=2)
        s0 = np.random.randn(8)
        g = np.zeros(8)
        acts = planner.plan_with_mcts(s0, g, horizon=4)
        self.assertIsInstance(acts, list)
        self.assertLessEqual(len(acts), 4)
        for a in acts:
            self.assertEqual(np.asarray(a).shape[0], 3)

    def test_maml_train_step_and_fast_adapt(self):
        maml = m.MAMLMetaLearner(model_dim=8, task_embedding_dim=4)
        support = [{"input": np.random.randn(8), "target": np.random.randn(8)} for _ in range(5)]
        query = [{"input": np.random.randn(8), "target": np.random.randn(8)} for _ in range(5)]
        maml.train_step([{"support": support, "query": query}])
        adapter = maml.fast_adapt("t1", support, num_steps=3)
        self.assertEqual(np.asarray(adapter).shape[0], 16)

    def test_hungarian_matching_is_injective(self):
        gr = m.GraphMatchingAnalogicalReasoning(node_dim=8)
        sim = np.random.randn(5, 3)
        matches = gr.hungarian_matching(sim)
        self.assertTrue(all(0 <= i < 5 and 0 <= j < 3 for i, j in matches))
        js = [j for _, j in matches]
        self.assertEqual(len(js), len(set(js)))

    def test_structural_causal_pc_graph_shape(self):
        scr = m.StructuralCausalReasoning(num_variables=5)
        for _ in range(80):
            scr.add_data(np.random.randn(5))
        g = scr.train_step(alpha=0.05)
        self.assertEqual(g.shape, (5, 5))
        self.assertTrue(np.all(np.diag(g) == 0))

    def test_grounded_dialogue_shapes_and_train(self):
        lang = m.GroundedSymbolicInterface(state_dim=8)
        utt = np.random.randn(128)
        goal = np.random.randn(128)
        resp = lang.multi_turn_dialogue([utt, utt], goal)
        self.assertEqual(len(resp), 2)
        self.assertEqual(np.asarray(resp[0]).shape[0], 128)

        batch = [(np.random.randn(128), np.random.randn(128), np.random.randn(128)) for _ in range(3)]
        loss = lang.train_step(batch, learning_rate=0.001)
        self.assertTrue(isinstance(loss, float) and np.isfinite(loss))

    def test_lifelong_ewc_train_step_runs(self):
        ll = m.LifelongLearningSystem(state_dim=8, action_dim=3)
        data = [(np.random.randn(8), np.random.randn(3)) for _ in range(20)]
        ll.update_fisher_information(data)
        ll.save_optimal_params()
        ll.train_step([(np.random.randn(8), np.random.randn(3)) for _ in range(10)], learning_rate=0.001)

    def test_facade_act_returns_16d_and_is_finite(self):
        _seed_all(0)
        ai = m.ActiveInferenceUpgradesFacade(state_dim=16, action_dim=16)
        o = np.random.randn(16)
        pref = np.random.randn(16)
        a = ai.act(o, goal=pref, horizon=3)
        self.assertEqual(np.asarray(a).shape, (16,))
        self.assertTrue(np.all(np.isfinite(a)))

    def test_canonical_policy_posterior_is_valid_distribution(self):
        _seed_all(0)
        ai = m.ActiveInferenceUpgradesFacade(state_dim=16, action_dim=16)
        # Trigger belief so canonical exists in a meaningful state.
        ai.act(np.random.randn(16), goal=np.zeros(16), horizon=2)
        G = np.array([0.0, 1.0, 2.0], dtype=float)
        qpi = ai.canonical.policy_posterior(G, context=np.zeros(16))
        self.assertEqual(qpi.shape, (3,))
        self.assertTrue(np.all(qpi >= 0.0))
        self.assertTrue(np.isfinite(np.sum(qpi)))
        self.assertAlmostEqual(float(np.sum(qpi)), 1.0, places=5)

    def test_brain_tick_emits_control_and_preferences(self):
        _seed_all(0)
        from mind.brain import AGIMind

        mind = AGIMind()
        # Ensure active inference exists in this repo configuration
        self.assertTrue(getattr(mind, 'active_inference', None) is not None)

        out = mind.tick(observation="hi", modality='text', reasoning_query="hello", reasoning_context=None, reward=0.0, done=False, learn=False, remember=False)
        self.assertIn('control', out)
        self.assertIsInstance(out['control'], dict)
        self.assertIn('recall_k', out['control'])
        self.assertIn('preferences', out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
