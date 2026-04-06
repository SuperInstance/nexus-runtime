"""Tests for agents.py — 45 tests."""

import math
import pytest
from jetson.rl.agents import (
    _softmax, _relu, _tanh, _sigm,
    TabularQLearning, DeepQLearning, PolicyGradientAgent, PPOLiteAgent,
)


class TestSoftmax:
    def test_basic(self):
        result = _softmax([1.0, 2.0, 3.0])
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-6

    def test_equal_inputs(self):
        result = _softmax([5.0, 5.0, 5.0, 5.0])
        assert all(abs(r - 0.25) < 1e-6 for r in result)

    def test_empty(self):
        assert _softmax([]) == []

    def test_single(self):
        result = _softmax([7.0])
        assert abs(result[0] - 1.0) < 1e-6

    def test_temperature(self):
        low_t = _softmax([1.0, 2.0], temperature=0.1)
        high_t = _softmax([1.0, 2.0], temperature=10.0)
        # Low temperature should be more peaked
        assert max(low_t) > max(high_t) or low_t[1] > high_t[1]


class TestRelu:
    def test_positive(self):
        assert _relu(5.0) == 5.0

    def test_negative(self):
        assert _relu(-3.0) == 0.0

    def test_zero(self):
        assert _relu(0.0) == 0.0


class TestTanh:
    def test_zero(self):
        assert abs(_tanh(0.0)) < 1e-10

    def test_large_positive(self):
        assert _tanh(100.0) > 0.99

    def test_large_negative(self):
        assert _tanh(-100.0) < -0.99


class TestSigm:
    def test_zero(self):
        assert abs(_sigm(0.0) - 0.5) < 1e-6

    def test_large_positive(self):
        assert _sigm(10.0) > 0.99

    def test_large_negative(self):
        assert _sigm(-10.0) < 0.01


class TestTabularQLearning:
    def setup_method(self):
        self.agent = TabularQLearning(n_actions=4, lr=0.1, gamma=0.99, epsilon=1.0)

    def test_create(self):
        assert self.agent.n_actions == 4
        assert self.agent.epsilon == 1.0

    def test_select_action_exploration(self):
        actions = set()
        for _ in range(100):
            actions.add(self.agent.select_action([0, 0, 0, 0]))
        # With epsilon=1.0, should try many different actions
        assert len(actions) > 1

    def test_update(self):
        self.agent.update([0, 0], 0, 1.0, [1, 0], False)
        q = self.agent.get_q_values([0, 0])
        assert q[0] != 0.0

    def test_update_terminal(self):
        self.agent.update([0, 0], 1, 10.0, [1, 1], True)
        q = self.agent.get_q_values([0, 0])
        assert q[1] > 0.0

    def test_decay_epsilon(self):
        eps_before = self.agent.epsilon
        self.agent.decay_epsilon()
        assert self.agent.epsilon < eps_before

    def test_epsilon_floor(self):
        self.agent.epsilon_min = 0.5
        for _ in range(100):
            self.agent.decay_epsilon()
        assert self.agent.epsilon >= 0.5

    def test_get_q_table(self):
        self.agent.update([1, 0], 0, 1.0, [1, 1], False)
        qt = self.agent.get_q_table()
        assert (1, 0) in qt

    def test_state_count(self):
        self.agent.update([0, 0], 0, 1.0, [0, 0], False)
        self.agent.update([1, 1], 0, 1.0, [1, 1], False)
        assert self.agent.get_state_count() == 2

    def test_greedy_after_decay(self):
        self.agent.epsilon = 0.0
        self.agent.update([0, 0], 2, 10.0, [0, 0], True)
        action = self.agent.select_action([0, 0])
        assert action == 2


class TestDeepQLearning:
    def setup_method(self):
        self.agent = DeepQLearning(state_dim=4, n_actions=2, lr=0.01)

    def test_create(self):
        assert self.agent.state_dim == 4
        assert self.agent.n_actions == 2

    def test_select_action(self):
        action = self.agent.select_action([0.0, 0.0, 0.0, 0.0])
        assert action in [0, 1]

    def test_update(self):
        self.agent.update([0, 0, 0, 0], 0, 1.0, [1, 0, 0, 0], False)
        w = self.agent.get_weights()
        assert len(w["b2"]) == 2

    def test_update_batch(self):
        batch = [([0, 0, 0, 0], 0, 1.0, [1, 0, 0, 0], False),
                 ([1, 0, 0, 0], 1, -1.0, [0, 0, 0, 0], True)]
        errors = self.agent.update_batch(batch)
        assert len(errors) == 2
        assert all(e >= 0 for e in errors)

    def test_target_update(self):
        self.agent.update([0, 0, 0, 0], 0, 1.0, [1, 0, 0, 0], False)
        self.agent.target_update()
        # Weights should be copied

    def test_decay_epsilon(self):
        self.agent.decay_epsilon()
        assert self.agent.epsilon < 1.0

    def test_get_step_count(self):
        self.agent.update([0, 0, 0, 0], 0, 1.0, [1, 0, 0, 0], False)
        assert self.agent.get_step_count() == 1

    def test_get_weights(self):
        w = self.agent.get_weights()
        assert "W1" in w
        assert "b1" in w
        assert "W2" in w
        assert "b2" in w

    def test_update_terminal(self):
        self.agent.update([0, 0, 0, 0], 0, 5.0, [0, 0, 0, 0], True)
        assert self.agent.get_step_count() == 1


class TestPolicyGradientAgent:
    def setup_method(self):
        self.agent = PolicyGradientAgent(state_dim=2, n_actions=3, lr=0.01)

    def test_create(self):
        assert self.agent.n_actions == 3

    def test_select_action(self):
        action = self.agent.select_action([1.0, 0.0])
        assert action in [0, 1, 2]

    def test_store_transition(self):
        self.agent.store_transition([0.0, 1.0], 1, 2.0)
        # No assertion needed beyond no crash

    def test_update_policy_no_transitions(self):
        loss = self.agent.update_policy()
        assert loss == 0.0

    def test_update_policy(self):
        for _ in range(5):
            self.agent.store_transition([1.0, 0.0], 0, 1.0)
        loss = self.agent.update_policy()
        assert loss > 0.0

    def test_get_policy_probs(self):
        probs = self.agent.get_policy_probs([1.0, 0.0])
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_clear_transitions(self):
        self.agent.store_transition([0, 0], 0, 1.0)
        self.agent.clear_transitions()
        loss = self.agent.update_policy()
        assert loss == 0.0

    def test_probs_sum_to_one_various_states(self):
        for state in [[0, 0], [1, 2], [-1, 3]]:
            probs = self.agent.get_policy_probs(state)
            assert abs(sum(probs) - 1.0) < 1e-6


class TestPPOLiteAgent:
    def setup_method(self):
        self.agent = PPOLiteAgent(state_dim=2, n_actions=3, lr=0.01)

    def test_create(self):
        assert self.agent.n_actions == 3
        assert self.agent.clip_ratio == 0.2

    def test_select_action(self):
        action = self.agent.select_action([1.0, 0.0])
        assert action in [0, 1, 2]

    def test_store_transition(self):
        probs = self.agent.get_policy_probs([0.0, 1.0])
        log_prob = math.log(max(probs[0], 1e-8))
        self.agent.store_transition([0, 1], 0, 1.0, 0.5, log_prob)

    def test_compute_advantages_empty(self):
        advs = self.agent.compute_advantages()
        assert advs == []

    def test_compute_advantages(self):
        probs = self.agent.get_policy_probs([0, 1])
        log_prob = math.log(max(probs[0], 1e-8))
        self.agent.store_transition([0, 1], 0, 1.0, 0.5, log_prob)
        self.agent.store_transition([1, 1], 1, 0.5, 0.3, log_prob)
        advs = self.agent.compute_advantages()
        assert len(advs) == 2

    def test_update_policy_batch_empty(self):
        loss = self.agent.update_policy_batch()
        assert loss == 0.0

    def test_update_policy_batch(self):
        probs = self.agent.get_policy_probs([0, 1])
        log_prob = math.log(max(probs[0], 1e-8))
        for _ in range(5):
            self.agent.store_transition([0, 1], 0, 1.0, 0.5, log_prob)
        loss = self.agent.update_policy_batch()
        assert loss > 0.0

    def test_update_value_empty(self):
        loss = self.agent.update_value()
        assert loss == 0.0

    def test_update_value(self):
        probs = self.agent.get_policy_probs([0, 1])
        log_prob = math.log(max(probs[0], 1e-8))
        for _ in range(5):
            self.agent.store_transition([0, 1], 0, 1.0, 0.5, log_prob)
        loss = self.agent.update_value()
        assert loss >= 0.0

    def test_get_value(self):
        val = self.agent.get_value([1.0, 0.0])
        assert isinstance(val, float)

    def test_buffer_size(self):
        assert self.agent.buffer_size() == 0
        probs = self.agent.get_policy_probs([0, 1])
        log_prob = math.log(max(probs[0], 1e-8))
        self.agent.store_transition([0, 1], 0, 1.0, 0.5, log_prob)
        assert self.agent.buffer_size() == 1

    def test_clear_buffer(self):
        probs = self.agent.get_policy_probs([0, 1])
        log_prob = math.log(max(probs[0], 1e-8))
        self.agent.store_transition([0, 1], 0, 1.0, 0.5, log_prob)
        self.agent.clear_buffer()
        assert self.agent.buffer_size() == 0

    def test_compute_returns(self):
        probs = self.agent.get_policy_probs([0, 1])
        log_prob = math.log(max(probs[0], 1e-8))
        self.agent.store_transition([0, 1], 0, 1.0, 0.5, log_prob)
        self.agent.store_transition([1, 1], 1, 0.5, 0.3, log_prob)
        rets = self.agent.compute_returns()
        assert len(rets) == 2
        assert rets[0] > rets[1]

    def test_get_policy_probs(self):
        probs = self.agent.get_policy_probs([1.0, 2.0])
        assert abs(sum(probs) - 1.0) < 1e-6
