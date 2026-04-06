"""Tests for path_planning.py — RRTStarPlanner, VoronoiDecomposer, ConsensusPlanner."""
import math, pytest
from jetson.swarm.path_planning import (
    Point, Obstacle, RRTStarConfig, RRTStarPlanner,
    VoronoiDecomposer, ConsensusPlanner
)

@pytest.fixture
def planner():
    return RRTStarPlanner(RRTStarConfig(
        max_iterations=300, step_size=5.0, goal_tolerance=2.0,
        rewire_radius=15.0, bias_factor=0.15
    ))

@pytest.fixture
def voronoi():
    return VoronoiDecomposer(bounds=(0, 0, 100, 100))

class TestPoint:
    def test_create(self):
        p = Point(x=1.0, y=2.0)
        assert p.x == 1.0 and p.y == 2.0
    def test_frozen(self):
        p = Point(x=0, y=0)
        with pytest.raises(AttributeError): p.x = 99

class TestRRTStarConfig:
    def test_defaults(self):
        c = RRTStarConfig()
        assert c.max_iterations == 500 and c.step_size == 5.0

class TestRRTStarPlan:
    def test_simple_path(self, planner):
        path = planner.plan((0, 0), (80, 80))
        assert len(path) >= 2
        assert path[0] == pytest.approx((0, 0), abs=1.0)
        assert path[-1] == pytest.approx((80, 80), abs=3.0)
    def test_short_distance(self, planner):
        path = planner.plan((0, 0), (5, 5))
        assert len(path) >= 2
    def test_with_obstacles(self, planner):
        obs = [Obstacle(x=40, y=40, radius=15)]
        path = planner.plan((0, 0), (80, 80), obstacles=obs)
        assert len(path) >= 2
    def test_with_bounds(self, planner):
        path = planner.plan((0, 0), (50, 50), bounds=(0, 0, 50, 50))
        assert len(path) >= 2
    def test_same_start_goal(self, planner):
        path = planner.plan((50, 50), (50, 50))
        assert len(path) >= 1

class TestRRTStarSmooth:
    def test_smooth_short(self, planner):
        path = [(0, 0), (10, 10), (20, 20)]
        smoothed = planner.smooth_path(path)
        assert len(smoothed) >= 1
    def test_smooth_empty(self, planner):
        assert planner.smooth_path([]) == []
    def test_smooth_single(self, planner):
        assert planner.smooth_path([(0, 0)]) == [(0, 0)]
    def test_smooth_with_obstacles(self, planner):
        path = [(0, 0), (25, 0), (50, 0), (75, 0), (100, 0)]
        obs = [Obstacle(x=50, y=10, radius=15)]
        smoothed = planner.smooth_path(path, obstacles=obs)
        assert len(smoothed) >= 1

class TestRRTStarMultiAgent:
    def test_two_agents(self, planner):
        paths = planner.plan_multi_agent(
            [(0, 0), (0, 80)],
            [(80, 80), (80, 0)],
        )
        assert len(paths) == 2
        assert len(paths[0]) >= 2
        assert len(paths[1]) >= 2
    def test_empty(self, planner):
        assert planner.plan_multi_agent([], []) == []

class TestRRTStarRewire:
    def test_rewire_method(self, planner):
        planner.plan((0, 0), (50, 50))
        planner.rewire(5, [])

class TestPointToSegment:
    def test_point_on_segment(self):
        d = RRTStarPlanner._point_to_segment_dist((0, 0), (10, 0), (5, 0))
        assert d == pytest.approx(0.0)
    def test_point_near_segment(self):
        d = RRTStarPlanner._point_to_segment_dist((0, 0), (10, 0), (5, 3))
        assert d == pytest.approx(3.0)
    def test_point_beyond_end(self):
        d = RRTStarPlanner._point_to_segment_dist((0, 0), (10, 0), (15, 0))
        assert d == pytest.approx(5.0)
    def test_degenerate_segment(self):
        d = RRTStarPlanner._point_to_segment_dist((5, 5), (5, 5), (0, 0))
        assert d == pytest.approx(math.hypot(5, 5))

class TestVoronoiDecompose:
    def test_single_agent(self, voronoi):
        cells = voronoi.decompose([(50, 50)], resolution=10.0)
        assert len(cells) == 1
        assert len(cells[0]) > 0
    def test_two_agents(self, voronoi):
        cells = voronoi.decompose([(25, 50), (75, 50)], resolution=10.0)
        assert len(cells) == 2
    def test_empty_agents(self, voronoi):
        assert voronoi.decompose([]) == {}
    def test_coverages(self, voronoi):
        cells = voronoi.decompose([(25, 50), (75, 50)], resolution=10.0)
        for pts in cells.values():
            assert len(pts) > 0

class TestVoronoiCentroids:
    def test_centroids_match_cells(self, voronoi):
        cells = voronoi.decompose([(25, 50), (75, 50)], resolution=10.0)
        centroids = voronoi.compute_centroids(cells)
        assert len(centroids) == 2
    def test_empty_cell(self):
        vd = VoronoiDecomposer()
        c = vd.compute_centroids({0: []})
        assert c[0] == (0.0, 0.0)

class TestVoronoiLloyd:
    def test_converges(self, voronoi):
        agents = [(30, 30), (70, 70)]
        relaxed = voronoi.lloyd_relaxation(agents, iterations=3, resolution=10.0)
        assert len(relaxed) == 2
    def test_single_agent(self, voronoi):
        relaxed = voronoi.lloyd_relaxation([(50, 50)], iterations=2, resolution=10.0)
        assert len(relaxed) == 1

class TestVoronoiArea:
    def test_total_area(self, voronoi):
        areas = voronoi.area_coverage([(25, 50), (75, 50)], resolution=10.0)
        total = sum(areas.values())
        assert total > 0

class TestConsensusPlanner:
    @pytest.fixture
    def cp(self):
        cp = ConsensusPlanner(agents=["a1", "a2", "a3"])
        return cp

    def test_add_agent(self, cp):
        cp.add_agent("a4")
        assert "a4" in cp.agents

    def test_remove_agent(self, cp):
        cp.remove_agent("a1")
        assert "a1" not in cp.agents

    def test_propose(self, cp):
        assert cp.propose_plan("a1", "p1", [(0, 0), (10, 10)]) is True

    def test_propose_unregistered(self, cp):
        assert cp.propose_plan("z1", "p1", [(0, 0)]) is False

    def test_vote(self, cp):
        cp.propose_plan("a1", "p1", [(0, 0), (10, 10)])
        cp.propose_plan("a2", "p1", [(0, 0), (20, 20)])
        assert cp.vote_on_proposals("p1", "a1", 0) is True

    def test_vote_invalid_plan(self, cp):
        assert cp.vote_on_proposals("p99", "a1", 0) is False

    def test_vote_invalid_index(self, cp):
        cp.propose_plan("a1", "p1", [(0, 0)])
        assert cp.vote_on_proposals("p1", "a1", 99) is False

    def test_resolve_majority(self, cp):
        cp.propose_plan("a1", "p1", [(0, 0)], priority=1)
        cp.propose_plan("a2", "p1", [(10, 10)], priority=2)
        cp.vote_on_proposals("p1", "a1", 0)
        cp.vote_on_proposals("p1", "a2", 0)
        cp.vote_on_proposals("p1", "a3", 0)
        result = cp.resolve_conflicts("p1")
        assert result is not None
        assert result["votes"] == 3

    def test_resolve_tiebreak_priority(self, cp):
        cp.propose_plan("a1", "p1", [(0, 0)], priority=5)
        cp.propose_plan("a2", "p1", [(10, 10)], priority=1)
        cp.vote_on_proposals("p1", "a1", 0)
        cp.vote_on_proposals("p1", "a2", 1)
        result = cp.resolve_conflicts("p1")
        assert result["proposal"]["priority"] == 5  # higher priority wins

    def test_resolve_no_proposals(self, cp):
        assert cp.resolve_conflicts("p99") is None

    def test_get_proposals(self, cp):
        cp.propose_plan("a1", "p1", [(0, 0)])
        assert len(cp.get_proposals("p1")) == 1

    def test_get_votes(self, cp):
        cp.propose_plan("a1", "p1", [(0, 0)])
        cp.vote_on_proposals("p1", "a1", 0)
        assert cp.get_votes("p1") == {"a1": 0}

    def test_clear_specific(self, cp):
        cp.propose_plan("a1", "p1", [(0, 0)])
        cp.clear("p1")
        assert cp.get_proposals("p1") == []

    def test_clear_all(self, cp):
        cp.propose_plan("a1", "p1", [(0, 0)])
        cp.propose_plan("a1", "p2", [(1, 1)])
        cp.clear()
        assert cp.get_proposals("p1") == []
        assert cp.get_proposals("p2") == []

    def test_get_current_plan_none(self, cp):
        assert cp.get_current_plan() is None

    def test_no_agents(self):
        cp = ConsensusPlanner()
        assert cp.propose_plan("a1", "p1", [(0, 0)]) is False
