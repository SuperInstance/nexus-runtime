"""Tests for replay_buffer.py — 40 tests."""

import pytest
from jetson.rl.replay_buffer import Transition, ReplayBuffer, PrioritizedReplayBuffer, EpisodeBuffer


class TestTransition:
    def test_create(self):
        t = Transition(state=[0, 0], action=1, reward=1.0, next_state=[1, 0], done=False)
        assert t.state == [0, 0]
        assert t.action == 1
        assert t.reward == 1.0
        assert t.next_state == [1, 0]
        assert t.done is False
        assert t.priority == 1.0

    def test_default_priority(self):
        t = Transition(state=0, action=0, reward=0.0, next_state=1, done=True)
        assert t.priority == 1.0

    def test_custom_priority(self):
        t = Transition(state=0, action=0, reward=0.0, next_state=1, done=True, priority=5.0)
        assert t.priority == 5.0

    def test_tuple_state(self):
        t = Transition(state=(1, 2), action=0, reward=-1.0, next_state=(2, 2), done=True)
        assert t.state == (1, 2)


class TestReplayBuffer:
    def setup_method(self):
        self.buf = ReplayBuffer(capacity=100)

    def test_create_empty(self):
        assert len(self.buf) == 0
        assert self.buf.is_empty()

    def test_add_single(self):
        self.buf.add([0, 0], 0, 1.0, [1, 0], False)
        assert len(self.buf) == 1
        assert not self.buf.is_empty()

    def test_add_many(self):
        for i in range(50):
            self.buf.add([i, 0], 0, float(i), [i + 1, 0], False)
        assert len(self.buf) == 50

    def test_fifo_eviction(self):
        for i in range(150):
            self.buf.add([i, 0], 0, 0.0, [i + 1, 0], False)
        assert len(self.buf) == 100  # capacity

    def test_sample(self):
        for i in range(50):
            self.buf.add([i, 0], 0, float(i), [i + 1, 0], False)
        batch = self.buf.sample(10)
        assert len(batch) == 10
        assert all(isinstance(t, Transition) for t in batch)

    def test_sample_larger_than_buffer(self):
        for i in range(5):
            self.buf.add([i], 0, 1.0, [i + 1], False)
        batch = self.buf.sample(10)
        assert len(batch) == 5

    def test_sample_empty(self):
        batch = self.buf.sample(5)
        assert len(batch) == 0

    def test_clear(self):
        self.buf.add([0], 0, 0.0, [1], False)
        self.buf.clear()
        assert len(self.buf) == 0

    def test_get_all(self):
        self.buf.add([0], 0, 0.0, [1], False)
        self.buf.add([1], 1, 1.0, [2], False)
        all_t = self.buf.get_all()
        assert len(all_t) == 2

    def test_can_sample(self):
        for i in range(3):
            self.buf.add([i], 0, 0.0, [i + 1], False)
        assert self.buf.can_sample(3) is True
        assert self.buf.can_sample(4) is False

    def test_add_transition(self):
        t = Transition(state=[5], action=2, reward=3.0, next_state=[6], done=True)
        self.buf.add_transition(t)
        assert len(self.buf) == 1

    def test_multiple_clears(self):
        self.buf.add([0], 0, 0.0, [1], False)
        self.buf.clear()
        self.buf.clear()
        assert len(self.buf) == 0

    def test_capacity_zero(self):
        buf = ReplayBuffer(capacity=0)
        buf.add([0], 0, 0.0, [1], False)
        assert len(buf) == 0


class TestPrioritizedReplayBuffer:
    def setup_method(self):
        self.buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

    def test_create_empty(self):
        assert len(self.buf) == 0
        assert self.buf.is_empty()

    def test_add_with_default_priority(self):
        self.buf.add([0, 0], 0, 1.0, [1, 0], False)
        assert len(self.buf) == 1

    def test_add_with_custom_priority(self):
        self.buf.add([0], 0, 1.0, [1], False, priority=10.0)
        priorities = self.buf.get_priorities()
        assert priorities[0] == 10.0

    def test_sample_returns_weights_and_indices(self):
        for i in range(10):
            self.buf.add([i], 0, float(i), [i + 1], False)
        transitions, weights, indices = self.buf.sample(5)
        assert len(transitions) == 5
        assert len(weights) == 5
        assert len(indices) == 5
        assert all(isinstance(w, float) for w in weights)

    def test_sample_empty(self):
        t, w, idx = self.buf.sample(5)
        assert t == []
        assert w == []
        assert idx == []

    def test_update_priorities(self):
        for i in range(5):
            self.buf.add([i], 0, 0.0, [i + 1], False)
        self.buf.update_priorities([0, 1, 2], [5.0, 3.0, 7.0])
        p = self.buf.get_priorities()
        assert p[0] == 5.0
        assert p[1] == 3.0
        assert p[2] == 7.0

    def test_anneal_beta(self):
        beta_before = self.buf.get_beta()
        self.buf.anneal_beta()
        assert self.buf.get_beta() > beta_before

    def test_beta_caps_at_1(self):
        for _ in range(2000):
            self.buf.anneal_beta()
        assert self.buf.get_beta() <= 1.0

    def test_fifo_eviction(self):
        for i in range(150):
            self.buf.add([i], 0, 0.0, [i + 1], False)
        assert len(self.buf) == 100

    def test_clear(self):
        self.buf.add([0], 0, 0.0, [1], False)
        self.buf.clear()
        assert len(self.buf) == 0
        assert self.buf.get_priorities() == []

    def test_get_all(self):
        for i in range(5):
            self.buf.add([i], 0, 0.0, [i + 1], False)
        all_t = self.buf.get_all()
        assert len(all_t) == 5

    def test_high_priority_more_likely(self):
        """Higher priority transitions should be sampled more often."""
        self.buf.add([0], 0, 0.0, [1], False, priority=100.0)
        self.buf.add([1], 0, 0.0, [2], False, priority=0.01)
        counts = {0: 0, 1: 0}
        for _ in range(200):
            _, _, indices = self.buf.sample(1)
            idx = indices[0]
            if idx == 0:
                counts[0] += 1
            elif idx == 1:
                counts[1] += 1
        # Index 0 has much higher priority, should be sampled more
        assert counts[0] > counts[1]


class TestEpisodeBuffer:
    def setup_method(self):
        self.buf = EpisodeBuffer(gamma=0.99, gae_lambda=0.95)

    def test_create_empty(self):
        assert self.buf.is_empty()
        assert len(self.buf) == 0

    def test_add_transition(self):
        self.buf.add([0], 0, 1.0, [1], False)
        assert self.buf.current_length() == 1
        assert self.buf.is_empty() is False

    def test_episode_finalize_on_done(self):
        self.buf.add([0], 0, 1.0, [1], True)
        assert self.buf.current_length() == 0
        assert self.buf.num_episodes() == 1

    def test_compute_returns(self):
        self.buf.add([0], 0, 1.0, [1], False)
        self.buf.add([1], 0, 1.0, [2], True)
        returns = self.buf.compute_returns()
        assert len(returns) == 0  # episode was finalized
        # Compute on stored episode
        ep = self.buf.get_episodes()[0]
        rets = self.buf.compute_returns(ep)
        assert len(rets) == 2
        assert abs(rets[0] - 1.0 - 0.99 * 1.0) < 1e-6
        assert abs(rets[1] - 1.0) < 1e-6

    def test_compute_gae(self):
        self.buf.add([0], 0, 1.0, [1], False)
        self.buf.add([1], 0, 1.0, [2], True)
        ep = self.buf.get_episodes()[0]
        values = [0.0, 0.0]
        gae = self.buf.compute_gae(values, ep)
        assert len(gae) == 2

    def test_gae_wrong_length(self):
        self.buf.add([0], 0, 1.0, [1], True)
        ep = self.buf.get_episodes()[0]
        gae = self.buf.compute_gae([0.0, 0.0, 0.0], ep)
        assert gae == []

    def test_finalize_episode(self):
        self.buf.add([0], 0, 1.0, [1], False)
        self.buf.finalize_episode()
        assert self.buf.num_episodes() == 1
        assert self.buf.current_length() == 0

    def test_get_episodes(self):
        self.buf.add([0], 0, 1.0, [1], True)
        eps = self.buf.get_episodes()
        assert len(eps) == 1
        assert len(eps[0]) == 1

    def test_clear(self):
        self.buf.add([0], 0, 1.0, [1], False)
        self.buf.clear()
        assert self.buf.is_empty()
        assert self.buf.num_episodes() == 0

    def test_add_transition_obj(self):
        t = Transition(state=[5], action=0, reward=2.0, next_state=[6], done=True)
        self.buf.add_transition(t)
        assert self.buf.num_episodes() == 1

    def test_multiple_episodes(self):
        for _ in range(3):
            self.buf.add([0], 0, 1.0, [1], True)
        assert self.buf.num_episodes() == 3

    def test_len_counts_all(self):
        self.buf.add([0], 0, 1.0, [1], True)
        self.buf.add([0], 0, 1.0, [1], False)
        assert len(self.buf) == 2

    def test_get_current_episode(self):
        self.buf.add([0], 0, 1.0, [1], False)
        curr = self.buf.get_current_episode()
        assert len(curr) == 1
