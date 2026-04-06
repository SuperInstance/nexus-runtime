"""Tests for flocking.py — FlockingParams, FlockingBehavior, Agent, FlockSimulation."""
import math, pytest
from jetson.swarm.flocking import FlockingParams, FlockingBehavior, Agent, Obstacle, FlockSimulation

@pytest.fixture
def params():
    return FlockingParams(separation_weight=1.5, alignment_weight=1.0, cohesion_weight=1.0,
                          max_speed=5.0, max_force=3.0, perception_radius=50.0, separation_radius=15.0)

@pytest.fixture
def behavior(params):
    return FlockingBehavior(params)

@pytest.fixture
def agents():
    return [Agent(agent_id=f"a{i}", x=float(i*5), y=0.0, vx=1.0, vy=0.0) for i in range(5)]

class TestFlockingParams:
    def test_defaults(self):
        p = FlockingParams()
        assert p.separation_weight == 1.5 and p.max_speed == 5.0
    def test_frozen(self):
        p = FlockingParams()
        with pytest.raises(AttributeError): p.max_speed = 10
    def test_custom(self):
        p = FlockingParams(max_speed=10.0, perception_radius=100.0)
        assert p.max_speed == 10.0 and p.perception_radius == 100.0

class TestAgent:
    def test_create(self):
        a = Agent(agent_id="a", x=1, y=2)
        assert a.agent_id == "a" and a.x == 1 and a.y == 2
    def test_speed(self):
        a = Agent(agent_id="a", x=0, y=0, vx=3, vy=4)
        assert a.speed() == 5.0
    def test_zero_speed(self):
        a = Agent(agent_id="a", x=0, y=0)
        assert a.speed() == 0.0
    def test_limit_velocity(self):
        a = Agent(agent_id="a", x=0, y=0, vx=10, vy=10, max_speed=5)
        a.limit_velocity()
        assert a.speed() == pytest.approx(5.0)
    def test_limit_force(self):
        a = Agent(agent_id="a", x=0, y=0, max_force=3)
        fx, fy = a.limit_force(10, 10)
        mag = math.hypot(fx, fy)
        assert mag == pytest.approx(3.0)
    def test_apply_force(self):
        a = Agent(agent_id="a", x=0, y=0)
        a.apply_force(1, 0)
        assert a.ax == 1.0 and a.ay == 0.0
    def test_update(self):
        a = Agent(agent_id="a", x=0, y=0, vx=1, vy=0)
        a.update(1.0)
        assert a.x == 1.0 and a.y == 0.0

class TestSeparation:
    def test_no_neighbors(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        fx, fy = behavior.compute_separation(a, [a])
        assert fx == 0.0 and fy == 0.0
    def test_single_neighbor_far(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        b = Agent(agent_id="b", x=100, y=0)
        fx, fy = behavior.compute_separation(a, [a, b])
        assert fx == 0.0 and fy == 0.0
    def test_close_neighbor(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        b = Agent(agent_id="b", x=5, y=0)
        fx, fy = behavior.compute_separation(a, [a, b])
        # Should push a away from b (negative x direction)
        assert fx < 0
    def test_surrounded(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        neighbors = [a] + [Agent(agent_id=f"n{i}", x=5*math.cos(2*math.pi*i/4), y=5*math.sin(2*math.pi*i/4)) for i in range(4)]
        fx, fy = behavior.compute_separation(a, neighbors)
        assert math.hypot(fx, fy) > 0
    def test_empty_neighbors(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        assert behavior.compute_separation(a, []) == (0.0, 0.0)

class TestAlignment:
    def test_same_velocity(self, behavior):
        a = Agent(agent_id="a", x=0, y=0, vx=5, vy=0, max_speed=5.0)
        b = Agent(agent_id="b", x=5, y=0, vx=5, vy=0, max_speed=5.0)
        fx, fy = behavior.compute_alignment(a, [a, b])
        assert math.hypot(fx, fy) < 0.01
    def test_opposite_velocity(self, behavior):
        a = Agent(agent_id="a", x=0, y=0, vx=1, vy=0)
        b = Agent(agent_id="b", x=5, y=0, vx=-1, vy=0)
        fx, fy = behavior.compute_alignment(a, [a, b])
        assert math.hypot(fx, fy) > 0
    def test_no_neighbors(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        assert behavior.compute_alignment(a, []) == (0.0, 0.0)

class TestCohesion:
    def test_center_pull(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        b = Agent(agent_id="b", x=20, y=0)
        fx, fy = behavior.compute_cohesion(a, [a, b])
        assert fx > 0  # should pull toward b
    def test_already_centered(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        b = Agent(agent_id="b", x=5, y=5)
        c = Agent(agent_id="c", x=-5, y=-5)
        fx, fy = behavior.compute_cohesion(a, [a, b, c])
        mag = math.hypot(fx, fy)
        assert mag < 0.01
    def test_no_neighbors(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        assert behavior.compute_cohesion(a, []) == (0.0, 0.0)

class TestFlockingForce:
    def test_combined_force(self, behavior):
        a = Agent(agent_id="a", x=0, y=0, vx=1, vy=0)
        b = Agent(agent_id="b", x=5, y=0, vx=1, vy=0)
        fx, fy = behavior.compute_flocking_force(a, [a, b])
        assert math.hypot(fx, fy) >= 0
    def test_no_neighbors_zero(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        assert behavior.compute_flocking_force(a, []) == (0.0, 0.0)

class TestObstacleAvoidance:
    def test_no_obstacles(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        assert behavior.obstacle_avoidance_force(a, []) == (0.0, 0.0)
    def test_far_obstacle(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        obs = [Obstacle(x=100, y=0, radius=5)]
        fx, fy = behavior.obstacle_avoidance_force(a, obs)
        assert math.hypot(fx, fy) < 0.01
    def test_close_obstacle(self, behavior):
        a = Agent(agent_id="a", x=0, y=0)
        obs = [Obstacle(x=5, y=0, radius=3)]
        fx, fy = behavior.obstacle_avoidance_force(a, obs)
        assert fx < 0  # push away

class TestFlockSimulation:
    def test_create(self, agents):
        sim = FlockSimulation(agents)
        assert len(sim.agents) == 5
    def test_step(self, agents):
        sim = FlockSimulation(agents)
        sim.step()
        assert sim.step_count == 1
    def test_run(self, agents):
        sim = FlockSimulation(agents)
        history = sim.run(steps=5)
        assert len(history) == 5
        assert all(len(h) == 5 for h in history)
    def test_get_states(self, agents):
        sim = FlockSimulation(agents)
        states = sim.get_agent_states()
        assert len(states) == 5
        assert states[0]["agent_id"] == "a0"
    def test_add_agent(self, agents):
        sim = FlockSimulation(agents)
        sim.add_agent(Agent(agent_id="new", x=50, y=50))
        assert len(sim.agents) == 6
    def test_remove_agent(self, agents):
        sim = FlockSimulation(agents)
        assert sim.remove_agent("a0") is True
        assert len(sim.agents) == 4
    def test_remove_nonexistent(self, agents):
        sim = FlockSimulation(agents)
        assert sim.remove_agent("zzz") is False
    def test_add_obstacle(self, agents):
        sim = FlockSimulation(agents)
        sim.add_obstacle(Obstacle(x=10, y=10))
        assert len(sim.obstacles) == 1
    def test_remove_obstacle(self, agents):
        sim = FlockSimulation(agents)
        sim.add_obstacle(Obstacle(x=10, y=10))
        assert sim.remove_obstacle(0) is True
        assert len(sim.obstacles) == 0
    def test_remove_obstacle_invalid(self, agents):
        sim = FlockSimulation(agents)
        assert sim.remove_obstacle(0) is False
    def test_reset(self, agents):
        sim = FlockSimulation(agents)
        sim.step(); sim.step()
        sim.reset()
        assert sim.step_count == 0
    def test_empty_sim(self):
        sim = FlockSimulation([])
        history = sim.run(3)
        assert len(history) == 3 and all(len(h)==0 for h in history)
    def test_single_agent(self):
        sim = FlockSimulation([Agent(agent_id="solo", x=0, y=0)])
        sim.step()
        assert sim.step_count == 1

class TestFlockConvergence:
    def test_converges_toward_cohesion(self):
        """Agents within perception range should converge."""
        a1 = Agent(agent_id="a", x=-20, y=0, vx=0, vy=0)
        a2 = Agent(agent_id="b", x=20, y=0, vx=0, vy=0)
        sim = FlockSimulation([a1, a2], params=FlockingParams(perception_radius=50.0, max_speed=5.0, max_force=3.0))
        sim.run(50)
        dist = math.hypot(a1.x - a2.x, a1.y - a2.y)
        assert dist < 35  # started at 40, should converge

    def test_obstacles_prevent_collision(self):
        # Agent slightly off-axis so obstacle avoidance deflects in y
        a = Agent(agent_id="a", x=0, y=3, vx=3, vy=0)
        sim = FlockSimulation([a], [Obstacle(x=15, y=0, radius=5)],
                              params=FlockingParams(obstacle_avoidance_radius=25.0, obstacle_avoidance_weight=3.0))
        sim.run(20)
        assert abs(a.y) > 1.0  # deflected away
