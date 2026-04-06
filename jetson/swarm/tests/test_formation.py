"""Tests for formation.py — FormationController, FormationType, VesselState."""
import math, pytest
from jetson.swarm.formation import FormationController, FormationError, FormationPosition, FormationType, VesselState

@pytest.fixture
def controller():
    return FormationController(FormationType.LINE, spacing=10.0, heading=0.0, center=(0.0, 0.0), error_threshold=2.0)

@pytest.fixture
def vessels_5():
    return [VesselState(vessel_id=f"v{i}", x=i*10, y=0.0, heading=0.0) for i in range(5)]

class TestFormationType:
    def test_enum_values_exist(self):
        assert len(FormationType) == 5
    def test_enum_unique(self):
        assert len(set(FormationType)) == len(FormationType)

class TestVesselState:
    def test_create(self):
        v = VesselState(vessel_id="a", x=1.0, y=2.0)
        assert v.vessel_id == "a" and v.x == 1.0 and v.y == 2.0
    def test_defaults(self):
        v = VesselState(vessel_id="b", x=0, y=0)
        assert v.heading == 0.0 and v.speed == 0.0 and v.active is True
    def test_frozen(self):
        v = VesselState(vessel_id="c", x=0, y=0)
        with pytest.raises(AttributeError): v.x = 99
    def test_inactive(self):
        v = VesselState(vessel_id="d", x=5, y=5, active=False)
        assert v.active is False

class TestComputePositions:
    def test_zero_count(self, controller): assert controller.compute_formation_positions(0) == []
    def test_one(self, controller):
        p = controller.compute_formation_positions(1)
        assert len(p) == 1 and p[0] == pytest.approx((0.0, 0.0))
    def test_three_centered(self, controller):
        p = controller.compute_formation_positions(3)
        assert len(p) == 3 and sum(x for x, _ in p) == pytest.approx(0.0, abs=1e-6)
    def test_spacing(self, controller):
        xs = sorted(x for x, _ in controller.compute_formation_positions(5))
        for i in range(1, len(xs)): assert xs[i]-xs[i-1] == pytest.approx(10.0)
    def test_custom_center(self, controller):
        ps = controller.compute_formation_positions(3, center=(100, 200))
        assert sum(x for x,_ in ps)/3 == pytest.approx(100, abs=1e-6)
        assert sum(y for _,y in ps)/3 == pytest.approx(200, abs=1e-6)

class TestFormationTypes:
    def test_wedge_one(self):
        c = FormationController(FormationType.WEDGE); assert len(c.compute_formation_positions(1)) == 1
    def test_wedge_lead_center(self):
        c = FormationController(FormationType.WEDGE, center=(50,50))
        assert c.compute_formation_positions(3)[0] == pytest.approx((50,50))
    def test_circle_one(self):
        c = FormationController(FormationType.CIRCLE)
        assert c.compute_formation_positions(1)[0] == pytest.approx((0,0))
    def test_circle_six(self):
        c = FormationController(FormationType.CIRCLE, spacing=10.0)
        ps = c.compute_formation_positions(6)
        dists = [math.hypot(x,y) for x,y in ps]
        assert max(dists)-min(dists) < 1.0
    def test_grid_four(self):
        c = FormationController(FormationType.GRID, spacing=10)
        ps = c.compute_formation_positions(4)
        assert len(ps) == 4
    def test_vshape_one(self):
        c = FormationController(FormationType.V_SHAPE)
        assert len(c.compute_formation_positions(1)) == 1
    def test_vshape_tip(self):
        c = FormationController(FormationType.V_SHAPE, center=(10,10))
        assert c.compute_formation_positions(5)[0] == pytest.approx((10,10))
    def test_heading_rotation(self):
        c = FormationController(FormationType.LINE, spacing=10, heading=math.pi/2)
        ps = c.compute_formation_positions(3)
        xs = [p[0] for p in ps]
        assert all(abs(x)<1e-6 for x in xs)

class TestAssign:
    def test_empty(self, controller): assert controller.assign_positions([]) == {}
    def test_all_inactive(self, controller):
        vs = [VesselState(f"v{i}", x=i, y=0, active=False) for i in range(3)]
        assert controller.assign_positions(vs) == {}
    def test_all_active(self, controller, vessels_5):
        assert len(controller.assign_positions(vessels_5)) == 5
    def test_skips_inactive(self, controller):
        vs = [VesselState("v0",0,0,active=True), VesselState("v1",10,0,active=True), VesselState("v2",20,0,active=False)]
        r = controller.assign_positions(vs)
        assert len(r)==2 and "v2" not in r
    def test_unique_slots(self, controller, vessels_5):
        r = controller.assign_positions(vessels_5)
        assert len({p.slot_index for p in r.values()}) == 5
    def test_custom_positions(self, controller):
        r = controller.assign_positions([VesselState("a",0,0)], positions=[(5,5)])
        assert r["a"].target_x == 5 and r["a"].target_y == 5

class TestErrors:
    def test_empty(self, controller): assert controller.compute_formation_errors([]) == []
    def test_unassigned_infinite(self, controller):
        errs = controller.compute_formation_errors([VesselState("x",100,100)])
        assert errs[0].position_error == float("inf") and errs[0].is_acceptable is False
    def test_on_target(self, controller):
        controller.assign_positions([VesselState("a",0,0)])
        e = controller.compute_formation_errors([VesselState("a",0,0)])[0]
        assert e.position_error < 0.01 and e.is_acceptable is True
    def test_off_target(self, controller):
        controller.assign_positions([VesselState("a",0,0)])
        e = controller.compute_formation_errors([VesselState("a",5,0)])[0]
        assert e.position_error > 4 and e.is_acceptable is False
    def test_skip_inactive(self, controller):
        assert controller.compute_formation_errors([VesselState("a",0,0,active=False)]) == []

class TestKeepalive:
    def test_single_false(self, controller):
        assert controller.formation_keepalive([VesselState("a",0,0)]) is False
    def test_empty_false(self, controller): assert controller.formation_keepalive([]) is False
    def test_all_on_target(self, controller):
        vs = [VesselState(f"v{i}", x=i*10, y=0) for i in range(3)]
        controller.assign_positions(vs)
        assert controller.formation_keepalive(vs) is True
    def test_far_off_false(self, controller):
        vs = [VesselState(f"v{i}", x=i*100, y=0) for i in range(3)]
        controller.assign_positions(vs)
        assert controller.formation_keepalive(vs) is False
    def test_counter_increments(self, controller):
        c = controller._keepalive_counter
        controller.formation_keepalive([VesselState("a",0,0),VesselState("b",10,0)])
        assert controller._keepalive_counter == c+1

class TestSwitching:
    def test_set_type(self, controller):
        controller.set_formation_type(FormationType.CIRCLE)
        assert controller.formation_type == FormationType.CIRCLE
    def test_history(self, controller):
        controller.set_formation_type(FormationType.WEDGE)
        controller.set_formation_type(FormationType.GRID)
        assert controller.get_formation_history() == [FormationType.LINE, FormationType.WEDGE]
    def test_initial_history(self, controller): assert controller.get_formation_history() == []

class TestReconfigure:
    def test_zero_active(self, controller):
        assert controller.reconfigure([VesselState("a",0,0,active=False)]) == {}
    def test_one_active(self, controller):
        assert controller.reconfigure([VesselState("a",0,0)]) == {}
    def test_two_switches_from_circle(self):
        c = FormationController(FormationType.CIRCLE)
        c.reconfigure([VesselState(f"v{i}",x=i*10,y=0) for i in range(2)])
        assert c.formation_type == FormationType.LINE
    def test_five_keeps_line(self, controller):
        vs = [VesselState(f"v{i}",x=i*10,y=0) for i in range(5)]
        assert len(controller.reconfigure(vs)) == 5

class TestEdgeCases:
    def test_large_formation(self, controller):
        assert len(controller.compute_formation_positions(100)) == 100
    def test_negative_spacing(self):
        c = FormationController(FormationType.LINE, spacing=-5)
        assert len(c.compute_formation_positions(3)) == 3
    def test_fractional_spacing(self):
        c = FormationController(FormationType.LINE, spacing=3.7)
        assert len(c.compute_formation_positions(4)) == 4
