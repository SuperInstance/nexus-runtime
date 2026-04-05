"""Tests for MQTT Topic Hierarchy — 13+ topics, parsing, generation."""

import pytest

from jetson.agent.mqtt_bridge.topics import (
    TopicPattern,
    TopicDefinition,
    TopicHierarchy,
    TopicParseResult,
    QoSLevel,
    TOPIC_DEFINITIONS,
    build_topic,
    parse_topic,
    vessel_topics,
    fleet_topics,
    all_vessel_subscriptions,
)


class TestTopicDefinitions:
    """Test topic definition completeness and correctness."""

    def test_thirteen_topic_patterns_defined(self):
        """All 13 topic patterns must be defined."""
        assert len(TopicPattern) >= 13

    def test_all_patterns_have_definitions(self):
        """Every TopicPattern must have a TopicDefinition."""
        for pattern in TopicPattern:
            assert pattern in TOPIC_DEFINITIONS, f"Missing definition for {pattern.name}"

    def test_vessel_topics_require_vessel_id(self):
        """Vessel-scoped topics must have {vessel_id} in their pattern."""
        vessel_patterns = TopicHierarchy.get_vessel_patterns()
        for pattern in vessel_patterns:
            assert "{vessel_id}" in pattern.value, (
                f"Vessel topic {pattern.name} missing {{vessel_id}} placeholder"
            )

    def test_fleet_topics_do_not_require_vessel_id(self):
        """Fleet-scoped topics must NOT have {vessel_id}."""
        fleet_patterns = TopicHierarchy.get_fleet_patterns()
        for pattern in fleet_patterns:
            assert "{vessel_id}" not in pattern.value, (
                f"Fleet topic {pattern.name} should not have {{vessel_id}}"
            )

    def test_qos_levels_correct(self):
        """QoS levels must match the Edge-Native spec."""
        assert TOPIC_DEFINITIONS[TopicPattern.TELEMETRY].qos == QoSLevel.AT_MOST_ONCE
        assert TOPIC_DEFINITIONS[TopicPattern.POSITION].qos == QoSLevel.AT_MOST_ONCE
        assert TOPIC_DEFINITIONS[TopicPattern.INTENTION].qos == QoSLevel.AT_MOST_ONCE
        assert TOPIC_DEFINITIONS[TopicPattern.COMMAND].qos == QoSLevel.AT_LEAST_ONCE
        assert TOPIC_DEFINITIONS[TopicPattern.STATUS].qos == QoSLevel.AT_LEAST_ONCE
        assert TOPIC_DEFINITIONS[TopicPattern.REFLEX_DEPLOY].qos == QoSLevel.EXACTLY_ONCE
        assert TOPIC_DEFINITIONS[TopicPattern.SAFETY_ALERT].qos == QoSLevel.EXACTLY_ONCE
        assert TOPIC_DEFINITIONS[TopicPattern.FLEET_SYNC].qos == QoSLevel.EXACTLY_ONCE

    def test_topic_names_unique(self):
        """All topic definition names must be unique."""
        names = [d.name for d in TOPIC_DEFINITIONS.values()]
        assert len(names) == len(set(names))

    def test_topic_prefix(self):
        """All topics must use 'nexus' prefix."""
        for pattern in TopicPattern:
            assert pattern.value.startswith("nexus/")

    def test_ten_vessel_patterns(self):
        """Must have 10 vessel-scoped patterns."""
        vessel = TopicHierarchy.get_vessel_patterns()
        assert len(vessel) == 10

    def test_three_fleet_patterns(self):
        """Must have 3 fleet-scoped patterns."""
        fleet = TopicHierarchy.get_fleet_patterns()
        assert len(fleet) == 3


class TestBuildTopic:
    """Test topic string generation."""

    def test_build_vessel_topic(self):
        """Build vessel-scoped topic with vessel_id."""
        topic = build_topic(TopicPattern.TELEMETRY, "vessel-001")
        assert topic == "nexus/vessel-001/telemetry"

    def test_build_all_vessel_topics(self):
        """Build all vessel topics for a vessel."""
        topics = vessel_topics("vessel-001")
        assert len(topics) == 10
        assert topics[TopicPattern.TELEMETRY] == "nexus/vessel-001/telemetry"
        assert topics[TopicPattern.STATUS] == "nexus/vessel-001/status"
        assert topics[TopicPattern.SAFETY_ALERT] == "nexus/vessel-001/safety/alert"
        assert topics[TopicPattern.COMMAND] == "nexus/vessel-001/command"
        assert topics[TopicPattern.POSITION] == "nexus/vessel-001/position"

    def test_build_fleet_topic(self):
        """Build fleet topic (no vessel_id needed)."""
        topic = build_topic(TopicPattern.FLEET_COORDINATION)
        assert topic == "nexus/fleet/coordination"

    def test_build_fleet_sync(self):
        topic = build_topic(TopicPattern.FLEET_SYNC)
        assert topic == "nexus/fleet/sync"

    def test_build_fleet_discovery(self):
        topic = build_topic(TopicPattern.FLEET_DISCOVERY)
        assert topic == "nexus/fleet/discovery"

    def test_build_vessel_topic_without_id_raises(self):
        """Building vessel topic without vessel_id must raise ValueError."""
        with pytest.raises(ValueError, match="requires a vessel_id"):
            build_topic(TopicPattern.TELEMETRY)

    def test_build_reflex_deploy_topic(self):
        topic = build_topic(TopicPattern.REFLEX_DEPLOY, "usv-alpha")
        assert topic == "nexus/usv-alpha/reflex/deploy"

    def test_build_reflex_result_topic(self):
        topic = build_topic(TopicPattern.REFLEX_RESULT, "usv-alpha")
        assert topic == "nexus/usv-alpha/reflex/result"

    def test_build_trust_events_topic(self):
        topic = build_topic(TopicPattern.TRUST_EVENTS, "usv-beta")
        assert topic == "nexus/usv-beta/trust/events"

    def test_build_intention_topic(self):
        topic = build_topic(TopicPattern.INTENTION, "vessel-42")
        assert topic == "nexus/vessel-42/intention"

    def test_build_response_topic(self):
        topic = build_topic(TopicPattern.RESPONSE, "vessel-42")
        assert topic == "nexus/vessel-42/response"


class TestParseTopic:
    """Test topic string parsing."""

    def test_parse_telemetry_topic(self):
        result = parse_topic("nexus/vessel-001/telemetry")
        assert result is not None
        assert result.pattern == TopicPattern.TELEMETRY
        assert result.vessel_id == "vessel-001"
        assert result.is_valid
        assert not result.fleet_topic

    def test_parse_safety_alert_topic(self):
        result = parse_topic("nexus/vessel-001/safety/alert")
        assert result is not None
        assert result.pattern == TopicPattern.SAFETY_ALERT
        assert result.vessel_id == "vessel-001"

    def test_parse_fleet_coordination(self):
        result = parse_topic("nexus/fleet/coordination")
        assert result is not None
        assert result.pattern == TopicPattern.FLEET_COORDINATION
        assert result.fleet_topic
        assert result.vessel_id is None

    def test_parse_fleet_sync(self):
        result = parse_topic("nexus/fleet/sync")
        assert result is not None
        assert result.pattern == TopicPattern.FLEET_SYNC

    def test_parse_fleet_discovery(self):
        result = parse_topic("nexus/fleet/discovery")
        assert result is not None
        assert result.pattern == TopicPattern.FLEET_DISCOVERY

    def test_parse_invalid_prefix(self):
        result = parse_topic("invalid/topic/here")
        assert result is None

    def test_parse_too_short(self):
        result = parse_topic("nexus")
        assert result is None

    def test_parse_unknown_vessel_topic(self):
        result = parse_topic("nexus/vessel-001/unknown_topic")
        assert result is not None
        assert result.pattern is None
        assert result.unknown
        assert result.vessel_id == "vessel-001"

    def test_parse_unknown_fleet_topic(self):
        result = parse_topic("nexus/fleet/unknown")
        assert result is not None
        assert result.pattern is None
        assert result.unknown
        assert result.fleet_topic

    def test_parse_returns_raw_topic(self):
        result = parse_topic("nexus/v-1/status")
        assert result.raw == "nexus/v-1/status"

    def test_parse_qos_on_valid_topic(self):
        result = parse_topic("nexus/v-1/telemetry")
        assert result.qos == QoSLevel.AT_MOST_ONCE

    def test_parse_qos_on_safety_topic(self):
        result = parse_topic("nexus/v-1/safety/alert")
        assert result.qos == QoSLevel.EXACTLY_ONCE

    def test_parse_category_vessel(self):
        result = parse_topic("nexus/v-1/command")
        assert result.category == "vessel"

    def test_parse_category_fleet(self):
        result = parse_topic("nexus/fleet/coordination")
        assert result.category == "fleet"


class TestTopicHierarchy:
    """Test TopicHierarchy class."""

    @pytest.fixture
    def hierarchy(self):
        return TopicHierarchy("vessel-001")

    def test_build_all_vessel_topics(self, hierarchy):
        vessel_patterns = TopicHierarchy.get_vessel_patterns()
        for pattern in vessel_patterns:
            topic = hierarchy.build(pattern)
            assert topic.startswith("nexus/vessel-001/")
            assert "{vessel_id}" not in topic

    def test_build_fleet_topic_raises_for_vessel_pattern(self, hierarchy):
        with pytest.raises(ValueError, match="requires a vessel_id"):
            hierarchy.build_fleet_topic(TopicPattern.TELEMETRY)

    def test_build_fleet_topic_ok(self, hierarchy):
        topic = hierarchy.build_fleet_topic(TopicPattern.FLEET_COORDINATION)
        assert topic == "nexus/fleet/coordination"

    def test_get_vessel_subscriptions(self, hierarchy):
        subs = hierarchy.get_vessel_subscriptions()
        assert len(subs) >= 7  # Inbound + bidirectional vessel topics

        topics_only = [t for t, q in subs]
        # Must include inbound topics
        assert any("command" in t for t in topics_only)
        assert any("reflex/deploy" in t for t in topics_only)
        # Must include fleet bidirectional topics
        assert any("fleet/coordination" in t for t in topics_only)
        assert any("fleet/discovery" in t for t in topics_only)
        assert any("fleet/sync" in t for t in topics_only)

    def test_get_publish_topics(self, hierarchy):
        pubs = hierarchy.get_publish_topics()
        assert len(pubs) >= 10
        topics_only = [t for t, q in pubs]
        # Must include outbound topics
        assert any("telemetry" in t for t in topics_only)
        assert any("safety/alert" in t for t in topics_only)
        assert any("position" in t for t in topics_only)

    def test_get_all_patterns(self):
        patterns = TopicHierarchy.get_all_patterns()
        assert len(patterns) >= 13

    def test_vessel_subscriptions_return_qos_tuples(self, hierarchy):
        subs = hierarchy.get_vessel_subscriptions()
        for topic, qos in subs:
            assert isinstance(topic, str)
            assert isinstance(qos, QoSLevel)
            assert topic.startswith("nexus/")


class TestAllVesselSubscriptions:
    """Test subscription generation for vessels."""

    def test_subscriptions_include_all_inbound(self):
        subs = all_vessel_subscriptions("v-1")
        topic_list = [t for t, q in subs]
        # command, reflex/deploy are inbound
        assert "nexus/v-1/command" in topic_list
        assert "nexus/v-1/reflex/deploy" in topic_list

    def test_subscriptions_include_fleet(self):
        subs = all_vessel_subscriptions("v-1")
        topic_list = [t for t, q in subs]
        assert "nexus/fleet/coordination" in topic_list
        assert "nexus/fleet/discovery" in topic_list
        assert "nexus/fleet/sync" in topic_list

    def test_different_vessels_different_topics(self):
        subs1 = all_vessel_subscriptions("v-1")
        subs2 = all_vessel_subscriptions("v-2")
        topics1 = [t for t, q in subs1]
        topics2 = [t for t, q in subs2]
        # Vessel-specific topics differ
        assert "nexus/v-1/command" in topics1
        assert "nexus/v-2/command" in topics2
        assert "nexus/v-1/command" not in topics2
        # Fleet topics are the same
        assert "nexus/fleet/coordination" in topics1
        assert "nexus/fleet/coordination" in topics2


class TestTopicParseResult:
    """Test TopicParseResult dataclass."""

    def test_valid_result_properties(self):
        result = TopicParseResult(
            pattern=TopicPattern.TELEMETRY,
            definition=TOPIC_DEFINITIONS[TopicPattern.TELEMETRY],
            vessel_id="v-1",
            fleet_topic=False,
            raw="nexus/v-1/telemetry",
        )
        assert result.is_valid
        assert result.qos == QoSLevel.AT_MOST_ONCE
        assert result.category == "vessel"

    def test_unknown_result_properties(self):
        result = TopicParseResult(
            pattern=None,
            definition=None,
            vessel_id="v-1",
            fleet_topic=False,
            raw="nexus/v-1/foobar",
            unknown=True,
        )
        assert not result.is_valid
        assert result.qos == QoSLevel.AT_MOST_ONCE  # default
        assert result.category == "vessel"
