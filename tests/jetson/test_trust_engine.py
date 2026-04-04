"""NEXUS Jetson tests - Trust engine tests."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "jetson"))

from trust_engine.increments import IncrementsEngine, IncrementsParams
from trust_engine.events import classify_event
from trust_engine.levels import get_level_definition, can_promote


class TestIncrementsEngine:
    """INCREMENTS trust engine tests."""

    def test_default_params(self) -> None:
        """Verify default parameter values."""
        params = IncrementsParams()
        assert params.alpha_gain == 0.002
        assert params.alpha_loss == 0.05
        assert params.t_floor == 0.2

    def test_engine_creation(self) -> None:
        """Verify engine can be created."""
        engine = IncrementsEngine()
        assert engine is not None

    def test_register_subsystem(self) -> None:
        """Verify subsystem registration."""
        engine = IncrementsEngine()
        engine.register_subsystem("steering")
        trust = engine.get_trust("steering")
        assert trust == 0.0

    def test_initial_autonomy(self) -> None:
        """Verify initial autonomy level is 0."""
        engine = IncrementsEngine()
        engine.register_subsystem("engine")
        level = engine.get_autonomy_level("engine")
        assert level == 0

    def test_deploy_blocked_low_trust(self) -> None:
        """Verify deployment blocked when trust is insufficient."""
        engine = IncrementsEngine()
        engine.register_subsystem("navigation")
        assert not engine.should_allow_deploy("navigation", 0.5)


class TestEventClassification:
    """Event classification tests."""

    def test_known_event(self) -> None:
        """Verify known event lookup."""
        event = classify_event("reflex_completed")
        assert event is not None
        assert event.category == "good"

    def test_unknown_event(self) -> None:
        """Verify unknown event returns None."""
        event = classify_event("nonexistent_event")
        assert event is None


class TestAutonomyLevels:
    """Autonomy level tests."""

    def test_level_definitions(self) -> None:
        """Verify all 6 levels are defined."""
        for level in range(6):
            defn = get_level_definition(level)
            assert defn is not None
            assert defn.level == level

    def test_no_promotion_at_zero(self) -> None:
        """Verify no promotion at zero trust."""
        assert can_promote(0.0, 0, 0.0, 0) == 0
