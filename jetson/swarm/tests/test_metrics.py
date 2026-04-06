"""Tests for metrics.py — SwarmMetrics, AgentSnapshot."""
import math, pytest
from jetson.swarm.metrics import AgentSnapshot, SwarmMetrics

@pytest.fixture
def metrics():
    return SwarmMetrics(comm_range=50.0, cohesion_threshold=30.0)

@pytest.fixture
def tight_cluster():
    return [
        AgentSnapshot(agent_id=f"a{i}", x=float(i), y=0.0, vx=1.0, vy=0.0, energy=100.0, connected=True)
        for i in range(5)
    ]

@pytest.fixture
def spread_out():
    return [
        AgentSnapshot(agent_id=f"a{i}", x=float(i*100), y=0.0, vx=1.0, vy=0.0, energy=80.0, connected=True)
        for i in range(5)
    ]

@pytest.fixture
def disconnected():
    return [
        AgentSnapshot(agent_id="a0", x=0, y=0, vx=1, vy=0, energy=100, connected=True),
        AgentSnapshot(agent_id="a1", x=200, y=200, vx=1, vy=0, energy=50, connected=False),
    ]

class TestAgentSnapshot:
    def test_create(self):
        a = AgentSnapshot(agent_id="x", x=1, y=2)
        assert a.agent_id == "x" and a.x == 1 and a.y == 2
    def test_defaults(self):
        a = AgentSnapshot(agent_id="x", x=0, y=0)
        assert a.vx == 0.0 and a.energy == 100.0 and a.connected is True
    def test_frozen(self):
        a = AgentSnapshot(agent_id="x", x=0, y=0)
        with pytest.raises(AttributeError): a.x = 99

class TestComputeSpread:
    def test_empty(self, metrics):
        assert metrics.compute_spread([]) == 0.0
    def test_single(self, metrics):
        assert metrics.compute_spread([AgentSnapshot("a", 0, 0)]) == 0.0
    def test_tight_cluster(self, metrics, tight_cluster):
        s = metrics.compute_spread(tight_cluster)
        assert s < 5.0
    def test_spread_out(self, metrics, spread_out):
        s = metrics.compute_spread(spread_out)
        assert s > 50.0

class TestComputeAlignment:
    def test_empty(self, metrics):
        assert metrics.compute_alignment([]) == 1.0
    def test_single(self, metrics):
        assert metrics.compute_alignment([AgentSnapshot("a", 0, 0)]) == 1.0
    def test_aligned(self, metrics):
        agents = [AgentSnapshot(f"a{i}", x=float(i*5), y=0, vx=2, vy=0) for i in range(5)]
        assert metrics.compute_alignment(agents) == pytest.approx(1.0, abs=0.01)
    def test_opposing(self, metrics):
        agents = [
            AgentSnapshot("a0", 0, 0, vx=2, vy=0),
            AgentSnapshot("a1", 5, 0, vx=-2, vy=0),
        ]
        a = metrics.compute_alignment(agents)
        assert a < 0.1
    def test_stationary(self, metrics):
        agents = [AgentSnapshot(f"a{i}", x=float(i), y=0) for i in range(5)]
        assert metrics.compute_alignment(agents) == 0.0

class TestComputeCohesion:
    def test_empty(self, metrics):
        assert metrics.compute_cohesion([]) == 1.0
    def test_single(self, metrics):
        assert metrics.compute_cohesion([AgentSnapshot("a", 0, 0)]) == 1.0
    def test_tight_high(self, metrics, tight_cluster):
        c = metrics.compute_cohesion(tight_cluster)
        assert c > 0.8
    def test_spread_low(self, metrics, spread_out):
        c = metrics.compute_cohesion(spread_out)
        assert c < 0.1

class TestComputeConnectivity:
    def test_empty(self, metrics):
        assert metrics.compute_connectivity([]) == 1.0
    def test_single(self, metrics):
        assert metrics.compute_connectivity([AgentSnapshot("a", 0, 0)]) == 1.0
    def test_connected_pair(self, metrics):
        agents = [AgentSnapshot("a0", 0, 0), AgentSnapshot("a1", 10, 0)]
        assert metrics.compute_connectivity(agents) == 1.0
    def test_disconnected_pair(self, metrics, disconnected):
        c = metrics.compute_connectivity(disconnected)
        assert c == 0.0  # only a0 connected, need 2+ connected
    def test_all_disconnected(self, metrics):
        agents = [AgentSnapshot(f"a{i}", x=float(i*100), y=0, connected=False) for i in range(3)]
        assert metrics.compute_connectivity(agents) == 0.0

class TestComputeEfficiency:
    def test_empty(self, metrics):
        assert metrics.compute_efficiency([]) == 0.0
    def test_perfect(self, metrics, tight_cluster):
        e = metrics.compute_efficiency(tight_cluster, 10, 10)
        assert e > 0.8
    def test_low_tasks(self, metrics, tight_cluster):
        e = metrics.compute_efficiency(tight_cluster, 1, 10)
        assert e < 0.8  # alignment+cohesion are high, task rate drags it down

class TestComputeRobustness:
    def test_empty(self, metrics):
        assert metrics.compute_robustness([]) == 0.0
    def test_high_robustness(self, metrics, tight_cluster):
        r = metrics.compute_robustness(tight_cluster)
        assert r > 0.8
    def test_low_robustness(self, metrics, disconnected):
        r = metrics.compute_robustness(disconnected)
        assert r < 1.0  # one agent disconnected reduces redundancy

class TestSwarmHealth:
    def test_empty(self, metrics):
        h = metrics.compute_swarm_health([])
        assert h["num_agents"] == 0
        assert h["spread"] == 0.0
    def test_all_keys_present(self, metrics, tight_cluster):
        h = metrics.compute_swarm_health(tight_cluster)
        for key in ["spread", "alignment", "cohesion", "connectivity", "efficiency", "robustness", "num_agents"]:
            assert key in h
    def test_num_agents(self, metrics):
        agents = [AgentSnapshot(f"a{i}", 0, 0) for i in range(7)]
        assert metrics.compute_swarm_health(agents)["num_agents"] == 7

class TestGenerateReport:
    def test_empty(self, metrics):
        r = metrics.generate_report([])
        assert r["status"] == "CRITICAL"
        assert r["overall_score"] == 0.0
    def test_healthy(self, metrics, tight_cluster):
        r = metrics.generate_report(tight_cluster, 10, 10)
        assert r["status"] in ["NOMINAL", "OPTIMAL"]
        assert "metrics" in r
        assert "recommendations" in r
    def test_degraded(self, metrics, spread_out):
        r = metrics.generate_report(spread_out, 0, 10)
        assert r["status"] in ["CRITICAL", "DEGRADED"]
    def test_recommendations_present(self, metrics, disconnected):
        r = metrics.generate_report(disconnected, 0, 5)
        assert len(r["recommendations"]) > 0
    def test_healthy_recommendations(self, metrics, tight_cluster):
        r = metrics.generate_report(tight_cluster, 10, 10)
        assert r["recommendations"] == ["Swarm is operating within normal parameters"]
