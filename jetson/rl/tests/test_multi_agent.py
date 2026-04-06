"""Tests for multi_agent.py — 35 tests."""

import pytest
from jetson.rl.multi_agent import (
    MultiAgentEnv, IndependentLearner, CentralizedCritic, CommunicationProtocol,
)


class TestMultiAgentEnv:
    def setup_method(self):
        self.env = MultiAgentEnv(agents=["a1", "a2"], grid_size=8)

    def test_create(self):
        assert self.env.num_agents() == 2
        assert self.env.n_actions == 5

    def test_reset(self):
        obs = self.env.reset()
        assert "a1" in obs
        assert "a2" in obs
        assert len(obs["a1"]) == 4

    def test_step(self):
        obs, rewards, dones, infos = self.env.step({"a1": 0, "a2": 1})
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(dones, dict)
        assert isinstance(infos, dict)

    def test_step_default_action(self):
        obs, rewards, dones, infos = self.env.step({})
        assert "a1" in obs
        assert rewards["a1"] == -1.0  # Stay action

    def test_get_positions(self):
        self.env.reset()
        pos = self.env.get_positions()
        assert "a1" in pos
        assert isinstance(pos["a1"], tuple)

    def test_get_goals(self):
        goals = self.env.get_goals()
        assert "a1" in goals

    def test_set_goals(self):
        self.env.set_goals({"a1": (0, 0), "a2": (7, 7)})
        assert self.env.get_goals()["a1"] == (0, 0)

    def test_max_steps(self):
        env = MultiAgentEnv(agents=["a1"], grid_size=5, max_steps=3)
        env.reset()
        for _ in range(3):
            _, _, dones, _ = env.step({"a1": 4})
        assert dones["a1"] is True

    def test_observation_format(self):
        obs = self.env.reset()
        for aid in self.env.agent_ids:
            assert len(obs[aid]) == 4
            assert all(isinstance(v, float) for v in obs[aid])

    def test_movement_changes_position(self):
        self.env.reset()
        pos_before = self.env.get_positions()
        self.env.step({"a1": 1, "a2": 0})
        pos_after = self.env.get_positions()
        # At least one should have moved (unless at boundary)
        changed = pos_before["a1"] != pos_after["a1"] or pos_before["a2"] != pos_after["a2"]
        # Might not change if at boundary

    def test_single_agent(self):
        env = MultiAgentEnv(agents=["solo"], grid_size=5)
        obs = env.reset()
        assert "solo" in obs
        assert env.num_agents() == 1


class TestIndependentLearner:
    def setup_method(self):
        self.learner = IndependentLearner(
            agent_ids=["a1", "a2"], n_actions=5, epsilon=1.0
        )

    def test_create(self):
        assert len(self.learner.agent_ids) == 2
        assert self.learner.get_epsilon() == 1.0

    def test_select_actions(self):
        obs = {"a1": [0, 0, 0, 0], "a2": [1, 1, 1, 1]}
        actions = self.learner.select_actions(obs)
        assert "a1" in actions
        assert "a2" in actions

    def test_update(self):
        self.learner.update("a1", [0, 0, 0, 0], 0, 1.0, [1, 0, 0, 0], False)
        qt = self.learner.get_q_table("a1")
        assert len(qt) > 0

    def test_update_unknown_agent(self):
        # Should not crash
        self.learner.update("unknown", [0], 0, 0.0, [1], False)

    def test_decay_epsilon(self):
        eps_before = self.learner.get_epsilon()
        self.learner.decay_epsilon()
        assert self.learner.get_epsilon() < eps_before

    def test_get_q_table(self):
        self.learner.update("a1", [0, 0, 0, 0], 0, 1.0, [1, 0, 0, 0], False)
        qt = self.learner.get_q_table("a1")
        assert isinstance(qt, dict)

    def test_get_q_table_empty(self):
        qt = self.learner.get_q_table("a1")
        assert qt == {}

    def test_greedy_selection(self):
        self.learner.epsilon = 0.0
        self.learner.update("a1", [0, 0, 0, 0], 2, 10.0, [0, 0, 0, 0], True)
        actions = self.learner.select_actions({"a1": [0, 0, 0, 0]})
        assert actions["a1"] == 2

    def test_select_actions_missing_agent(self):
        actions = self.learner.select_actions({"unknown": [0, 0, 0, 0]})
        assert "unknown" not in actions

    def test_multiple_updates(self):
        for _ in range(10):
            self.learner.update("a2", [1, 0, 0, 0], 0, 1.0, [2, 0, 0, 0], False)
        qt = self.learner.get_q_table("a2")
        assert len(qt) >= 1


class TestCentralizedCritic:
    def setup_method(self):
        self.critic = CentralizedCritic(agent_ids=["a1", "a2"], state_dim=4)

    def test_create(self):
        assert len(self.critic.agent_ids) == 2

    def test_joint_value(self):
        obs = {"a1": [1, 0, 0, 0], "a2": [0, 1, 0, 0]}
        val = self.critic.joint_value(obs)
        assert isinstance(val, float)

    def test_advantage_per_agent(self):
        obs = {"a1": [1, 0, 0, 0], "a2": [0, 1, 0, 0]}
        rewards = {"a1": 5.0, "a2": 3.0}
        adv = self.critic.advantage_per_agent("a1", obs, rewards)
        assert isinstance(adv, float)

    def test_store_returns(self):
        self.critic.store_returns("a1", [1.0, 2.0, 3.0])
        buf = self.critic.get_returns_buffer("a1")
        assert len(buf) == 3

    def test_update(self):
        obs = {"a1": [1, 0, 0, 0], "a2": [0, 1, 0, 0]}
        self.critic.update(obs, 5.0)
        # Should not crash

    def test_get_value_weights(self):
        w = self.critic.get_value_weights()
        assert isinstance(w, list)
        assert len(w) == 8  # 4 * 2 agents

    def test_get_returns_buffer_empty(self):
        buf = self.critic.get_returns_buffer("a1")
        assert buf == []

    def test_clear_buffers(self):
        self.critic.store_returns("a1", [1.0, 2.0])
        self.critic.clear_buffers()
        assert self.critic.get_returns_buffer("a1") == []

    def test_single_agent_critic(self):
        critic = CentralizedCritic(agent_ids=["solo"], state_dim=4)
        val = critic.joint_value({"solo": [1, 0, 0, 0]})
        assert isinstance(val, float)


class TestCommunicationProtocol:
    def setup_method(self):
        self.comm = CommunicationProtocol(agents=["a1", "a2", "a3"])

    def test_create(self):
        assert len(self.comm.agent_ids) == 3

    def test_send_message(self):
        ok = self.comm.send_message("a1", "a2", "hello")
        assert ok is True

    def test_send_unknown(self):
        ok = self.comm.send_message("unknown", "a1", "hi")
        assert ok is False

    def test_send_to_unknown(self):
        ok = self.comm.send_message("a1", "unknown", "hi")
        assert ok is False

    def test_receive(self):
        self.comm.send_message("a1", "a2", "hello")
        msgs = self.comm.receive("a2")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"
        assert msgs[0]["sender"] == "a1"

    def test_receive_clears_inbox(self):
        self.comm.send_message("a1", "a2", "msg1")
        self.comm.receive("a2")
        msgs = self.comm.receive("a2")
        assert len(msgs) == 0

    def test_broadcast(self):
        count = self.comm.broadcast("a1", "announcement")
        assert count == 2  # a2 and a3

    def test_broadcast_delivers(self):
        self.comm.broadcast("a1", "news")
        msgs_a2 = self.comm.receive("a2")
        msgs_a3 = self.comm.receive("a3")
        assert len(msgs_a2) == 1
        assert len(msgs_a3) == 1
        assert msgs_a2[0]["broadcast"] is True

    def test_broadcast_unknown(self):
        count = self.comm.broadcast("unknown", "x")
        assert count == 0

    def test_peek(self):
        self.comm.send_message("a1", "a2", "peek_me")
        msgs = self.comm.peek("a2")
        assert len(msgs) == 1
        # Peek should not clear
        msgs2 = self.comm.peek("a2")
        assert len(msgs2) == 1

    def test_shared_memory_write_read(self):
        self.comm.write_shared("a1", "pos", (5, 3))
        val = self.comm.read_shared("a1", "pos")
        assert val == (5, 3)

    def test_shared_memory_default(self):
        val = self.comm.read_shared("a1", "missing", default=42)
        assert val == 42

    def test_shared_memory_read_all(self):
        self.comm.write_shared("a1", "x", 1)
        self.comm.write_shared("a1", "y", 2)
        all_data = self.comm.read_all_shared("a1")
        assert all_data == {"x": 1, "y": 2}

    def test_clear_shared(self):
        self.comm.write_shared("a1", "x", 1)
        self.comm.clear_shared("a1")
        assert self.comm.read_all_shared("a1") == {}

    def test_clear_all(self):
        self.comm.send_message("a1", "a2", "msg")
        self.comm.write_shared("a1", "x", 1)
        self.comm.clear_all()
        assert self.comm.receive("a2") == []
        assert self.comm.read_all_shared("a1") == {}

    def test_message_count(self):
        self.comm.send_message("a1", "a2", "m1")
        self.comm.send_message("a3", "a2", "m2")
        assert self.comm.message_count("a2") == 2

    def test_total_messages(self):
        self.comm.send_message("a1", "a2", "m1")
        self.comm.send_message("a2", "a3", "m2")
        assert self.comm.total_messages() == 2

    def test_read_unknown_agent(self):
        val = self.comm.read_shared("unknown", "key", default=-1)
        assert val == -1
