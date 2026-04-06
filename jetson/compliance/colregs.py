"""COLREGs (International Regulations for Preventing Collisions at Sea) Rule Verification.

Implements verification of navigation rules for marine autonomous vessels
including Rules 5-19 (steering and sailing), Rules 20-32 (lights and shapes),
and Rules 34-37 (sound signals).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import math


class VisibilityCondition(Enum):
    """Navigation visibility conditions."""
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    RESTRICTED = "restricted"


class WaterwayType(Enum):
    """Types of waterways."""
    OPEN_SEA = "open_sea"
    NARROW_CHANNEL = "narrow_channel"
    TRAFFIC_SEPARATION = "traffic_separation"
    INLAND = "inland"
    PORT = "port"


class VesselType(Enum):
    """Types of vessels per COLREGs."""
    POWER_DRIVEN = "power_driven"
    SAILING = "sailing"
    FISHING = "fishing"
    NOT_UNDER_COMMAND = "not_under_command"
    RESTRICTED_MANEUVERABILITY = "restricted_maneuverability"
    CONSTRAINED_BY_DRAFT = "constrained_by_draft"
    ENGAGED_IN_FISHING = "engaged_in_fishing"
    PILOT_VESSEL = "pilot_vessel"
    ANCHORED = "anchored"
    AGROUND = "aground"
    ASV = "asv"  # Autonomous surface vessel


class RuleNumber(Enum):
    """COLREGs rule numbers."""
    RULE_5 = "Rule 5"
    RULE_6 = "Rule 6"
    RULE_7 = "Rule 7"
    RULE_8 = "Rule 8"
    RULE_9 = "Rule 9"
    RULE_10 = "Rule 10"
    RULE_11 = "Rule 11"
    RULE_12 = "Rule 12"
    RULE_13 = "Rule 13"
    RULE_14 = "Rule 14"
    RULE_15 = "Rule 15"
    RULE_16 = "Rule 16"
    RULE_17 = "Rule 17"
    RULE_18 = "Rule 18"
    RULE_19 = "Rule 19"


class UrgencyLevel(Enum):
    """Urgency levels for COLREGs actions."""
    IMMEDIATE = "immediate"
    URGENT = "urgent"
    CAUTIONARY = "cautionary"
    INFORMATIONAL = "informational"


@dataclass
class VesselState:
    """State of a single vessel."""
    vessel_type: VesselType
    heading: float  # degrees 0-360
    speed: float  # knots
    position: Tuple[float, float]  # (latitude, longitude) or (x, y)
    length: float = 10.0  # meters
    beam: float = 3.0  # meters
    name: str = ""
    has_night_lights: bool = True
    has_sound_signals: bool = True
    restricted: bool = False


@dataclass
class VesselSituation:
    """Complete navigation situation."""
    own_vessel: VesselState
    other_vessels: List[VesselState] = field(default_factory=list)
    visibility: VisibilityCondition = VisibilityCondition.GOOD
    waterway_type: WaterwayType = WaterwayType.OPEN_SEA
    time_of_day: str = "day"  # "day" or "night"


@dataclass
class RuleReference:
    """Reference to a COLREGs rule."""
    rule_number: str
    title: str
    description: str
    applicable_when: str


@dataclass
class COLREGsResult:
    """Result of COLREGs rule checking."""
    applicable_rules: List[RuleReference] = field(default_factory=list)
    recommended_action: str = ""
    urgency: UrgencyLevel = UrgencyLevel.INFORMATIONAL
    rule_references: List[str] = field(default_factory=list)
    give_way_vessel: str = ""
    stand_on_vessel: str = ""
    reasoning: str = ""


class COLREGsChecker:
    """COLREGs rule verification engine for autonomous marine vessels."""

    def __init__(self) -> None:
        self._rules = self._build_rules_db()

    def _build_rules_db(self) -> Dict[str, RuleReference]:
        """Build the COLREGs rules database."""
        return {
            "Rule 5": RuleReference(
                rule_number="Rule 5",
                title="Look-out",
                description="Every vessel shall at all times maintain a proper look-out by sight and hearing as well as by all available means appropriate.",
                applicable_when="At all times, every vessel",
            ),
            "Rule 6": RuleReference(
                rule_number="Rule 6",
                title="Safe Speed",
                description="Every vessel shall at all times proceed at a safe speed so that she can take proper and effective action to avoid collision.",
                applicable_when="At all times, every vessel",
            ),
            "Rule 7": RuleReference(
                rule_number="Rule 7",
                title="Risk of Collision",
                description="Every vessel shall use all available means appropriate to the prevailing circumstances and conditions to determine if risk of collision exists.",
                applicable_when="When risk of collision may exist",
            ),
            "Rule 8": RuleReference(
                rule_number="Rule 8",
                title="Action to Avoid Collision",
                description="Any action taken to avoid collision shall be taken in accordance with the Rules of this Part and shall be positive, made in ample time and with due regard to the observance of good seamanship.",
                applicable_when="When collision risk exists",
            ),
            "Rule 9": RuleReference(
                rule_number="Rule 9",
                title="Narrow Channels",
                description="A vessel proceeding along the course of a narrow channel shall keep as near to the outer limit of the channel which lies on her starboard side as is safe and practicable.",
                applicable_when="In narrow channels",
            ),
            "Rule 10": RuleReference(
                rule_number="Rule 10",
                title="Traffic Separation Schemes",
                description="A vessel using a traffic separation scheme shall proceed in the appropriate traffic lane in the general direction of traffic flow for that lane.",
                applicable_when="In traffic separation schemes",
            ),
            "Rule 11": RuleReference(
                rule_number="Rule 11",
                title="Application",
                description="Rules in this section apply to vessels in sight of one another.",
                applicable_when="When vessels are in sight of one another",
            ),
            "Rule 12": RuleReference(
                rule_number="Rule 12",
                title="Sailing Vessels",
                description="When two sailing vessels are approaching one another, so as to involve risk of collision, one of them shall keep out of the way of the other.",
                applicable_when="When two sailing vessels approach with collision risk",
            ),
            "Rule 13": RuleReference(
                rule_number="Rule 13",
                title="Overtaking",
                description="Notwithstanding anything contained in the Rules, any vessel overtaking any other shall keep out of the way of the vessel being overtaken.",
                applicable_when="When one vessel is overtaking another",
            ),
            "Rule 14": RuleReference(
                rule_number="Rule 14",
                title="Head-on Situation",
                description="When two power-driven vessels are meeting on reciprocal or nearly reciprocal courses so as to involve risk of collision each shall alter her course to starboard so that each shall pass on the port side of the other.",
                applicable_when="When two power-driven vessels are meeting head-on",
            ),
            "Rule 15": RuleReference(
                rule_number="Rule 15",
                title="Crossing Situation",
                description="When two power-driven vessels are crossing so as to involve risk of collision, the vessel which has the other on her own starboard side shall keep out of the way.",
                applicable_when="When two power-driven vessels are crossing",
            ),
            "Rule 16": RuleReference(
                rule_number="Rule 16",
                title="Action by Give-way Vessel",
                description="Every vessel which is directed to keep out of the way of another vessel shall, so far as possible, take early and substantial action to keep well clear.",
                applicable_when="When vessel is the give-way vessel",
            ),
            "Rule 17": RuleReference(
                rule_number="Rule 17",
                title="Action by Stand-on Vessel",
                description="Where one of two vessels is to keep out of the way, the other shall keep her course and speed.",
                applicable_when="When vessel is the stand-on vessel",
            ),
            "Rule 18": RuleReference(
                rule_number="Rule 18",
                title="Responsibilities Between Vessels",
                description="A power-driven vessel underway shall keep out of the way of: a vessel not under command, a vessel restricted in her ability to maneuver, a vessel engaged in fishing, a sailing vessel.",
                applicable_when="Between vessels of different types",
            ),
            "Rule 19": RuleReference(
                rule_number="Rule 19",
                title="Conduct of Vessels in Restricted Visibility",
                description="Every vessel shall proceed at a safe speed adapted to the prevailing circumstances and conditions of restricted visibility. A power-driven vessel shall have her engines ready for immediate manoeuvre.",
                applicable_when="When visibility is restricted",
            ),
            "Rule 20": RuleReference(
                rule_number="Rule 20",
                title="Application of Lights and Shapes",
                description="Rules in this Part shall be complied with in all weathers from sunset to sunrise, and during such times no other lights shall be exhibited.",
                applicable_when="From sunset to sunrise, and in restricted visibility",
            ),
            "Rule 25": RuleReference(
                rule_number="Rule 25",
                title="Sailing Vessels and Vessels Under Oars",
                description="A sailing vessel underway shall exhibit sidelights and a sternlight. A vessel under oars may exhibit those lights but shall, if she does not, have ready at hand an electric torch or lighted lantern.",
                applicable_when="Sailing vessels underway at night",
            ),
            "Rule 34": RuleReference(
                rule_number="Rule 34",
                title="Manoeuvring and Warning Signals",
                description="When vessels are in sight of one another, a power-driven vessel underway, when manoeuvring shall indicate her manoeuvre by sound signals.",
                applicable_when="When manoeuvring in sight of another vessel",
            ),
            "Rule 35": RuleReference(
                rule_number="Rule 35",
                title="Sound Signals in Restricted Visibility",
                description="In or near an area of restricted visibility, a power-driven vessel making way through the water shall sound at intervals of not more than 2 minutes one prolonged blast.",
                applicable_when="In restricted visibility",
            ),
            "Rule 37": RuleReference(
                rule_number="Rule 37",
                title="Distress Signals",
                description="When a vessel is in distress and requires assistance she shall use or exhibit the signals in Annex IV.",
                applicable_when="When a vessel is in distress",
            ),
        }

    def _get_rule(self, rule_number: str) -> Optional[RuleReference]:
        """Get a rule reference by number."""
        return self._rules.get(rule_number)

    def _bearing_to(self, from_vessel: VesselState, to_vessel: VesselState) -> float:
        """Calculate relative bearing from one vessel to another (degrees).

        Returns bearing relative to own vessel's heading (0 = ahead).
        """
        dx = to_vessel.position[0] - from_vessel.position[0]
        dy = to_vessel.position[1] - from_vessel.position[1]
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return 0.0
        absolute_bearing = math.degrees(math.atan2(dx, dy)) % 360
        relative_bearing = (absolute_bearing - from_vessel.heading) % 360
        return relative_bearing

    def _distance_to(self, from_vessel: VesselState, to_vessel: VesselState) -> float:
        """Calculate distance between two vessels."""
        dx = to_vessel.position[0] - from_vessel.position[0]
        dy = to_vessel.position[1] - from_vessel.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def _is_approaching(self, own: VesselState, other: VesselState) -> bool:
        """Check if two vessels are approaching each other."""
        dist = self._distance_to(own, other)
        if dist < 1e-10:
            return False
        # Simple check: if bearing is within ±45° of ahead, consider approaching
        bearing = self._bearing_to(own, other)
        return (bearing < 45 or bearing > 315) and dist < 1000

    def get_applicable_rules(self, situation: VesselSituation) -> List[RuleReference]:
        """Get all applicable rules for a given situation.

        Args:
            situation: Current navigation situation

        Returns:
            List of applicable RuleReference objects
        """
        rules = []

        # Always applicable rules
        rules.append(self._get_rule("Rule 5"))
        rules.append(self._get_rule("Rule 6"))
        rules.append(self._get_rule("Rule 7"))
        rules.append(self._get_rule("Rule 8"))

        # Waterway-specific rules
        if situation.waterway_type == WaterwayType.NARROW_CHANNEL:
            rules.append(self._get_rule("Rule 9"))
        elif situation.waterway_type == WaterwayType.TRAFFIC_SEPARATION:
            rules.append(self._get_rule("Rule 10"))

        # Visibility-dependent rules
        if situation.visibility in (VisibilityCondition.RESTRICTED, VisibilityCondition.POOR):
            rules.append(self._get_rule("Rule 19"))

        # Vessel interaction rules
        for other in situation.other_vessels:
            if self._is_approaching(situation.own_vessel, other):
                rules.append(self._get_rule("Rule 13"))  # Overtaking check
                rules.append(self._get_rule("Rule 14"))  # Head-on check
                rules.append(self._get_rule("Rule 15"))  # Crossing check
                rules.append(self._get_rule("Rule 16"))  # Give-way action
                rules.append(self._get_rule("Rule 17"))  # Stand-on action
                rules.append(self._get_rule("Rule 18"))  # Responsibilities

        # Night time rules
        if situation.time_of_day == "night":
            rules.append(self._get_rule("Rule 20"))

        return [r for r in rules if r is not None]

    def check_head_on_situation(self, situation: VesselSituation) -> COLREGsResult:
        """Check for head-on situations (Rule 14).

        Two power-driven vessels meeting on reciprocal or nearly reciprocal courses.

        Args:
            situation: Current navigation situation

        Returns:
            COLREGsResult with head-on assessment
        """
        rule = self._get_rule("Rule 14")
        results = []

        for other in situation.other_vessels:
            bearing = self._bearing_to(situation.own_vessel, other)
            dist = self._distance_to(situation.own_vessel, other)

            # Head-on: other vessel is within ±10° of own heading and approaching
            if dist < 1000 and (bearing < 10 or bearing > 350):
                if self._is_approaching(situation.own_vessel, other):
                    urgency = UrgencyLevel.IMMEDIATE if dist < 200 else UrgencyLevel.URGENT
                    results.append(COLREGsResult(
                        applicable_rules=[rule] if rule else [],
                        recommended_action=(
                            "Alter course to starboard so that each vessel "
                            "passes on the port side of the other"
                        ),
                        urgency=urgency,
                        rule_references=["Rule 14"],
                        give_way_vessel="Both vessels (each alter to starboard)",
                        stand_on_vessel="",
                        reasoning=(
                            f"Head-on situation detected: vessel '{other.name}' "
                            f"at bearing {bearing:.1f}°, distance {dist:.0f}m. "
                            f"Rule 14 applies — both vessels alter course to starboard."
                        ),
                    ))

        if results:
            return max(results, key=lambda r: (
                0 if r.urgency == UrgencyLevel.IMMEDIATE
                else 1 if r.urgency == UrgencyLevel.URGENT
                else 2
            ))

        return COLREGsResult(
            applicable_rules=[],
            recommended_action="No head-on situation detected",
            urgency=UrgencyLevel.INFORMATIONAL,
            rule_references=[],
        )

    def check_crossing_situation(self, situation: VesselSituation) -> COLREGsResult:
        """Check for crossing situations (Rule 15).

        When two power-driven vessels are crossing, the vessel with the other
        on her starboard side shall give way.

        Args:
            situation: Current navigation situation

        Returns:
            COLREGsResult with crossing assessment
        """
        rule = self._get_rule("Rule 15")
        results = []

        for other in situation.other_vessels:
            bearing = self._bearing_to(situation.own_vessel, other)
            dist = self._distance_to(situation.own_vessel, other)

            # Crossing: other vessel is 10-170° on starboard side (0-170 is starboard)
            if dist < 1000 and self._is_approaching(situation.own_vessel, other):
                if 10 <= bearing <= 170:
                    # Other vessel on our starboard — we give way
                    urgency = UrgencyLevel.URGENT if dist < 500 else UrgencyLevel.CAUTIONARY
                    results.append(COLREGsResult(
                        applicable_rules=[rule] if rule else [],
                        recommended_action=(
                            f"Give way to '{other.name}' by altering course to starboard "
                            f"and/or reducing speed"
                        ),
                        urgency=urgency,
                        rule_references=["Rule 15", "Rule 16"],
                        give_way_vessel=situation.own_vessel.name or "own vessel",
                        stand_on_vessel=other.name,
                        reasoning=(
                            f"Crossing situation: vessel '{other.name}' "
                            f"at bearing {bearing:.1f}° (starboard side), "
                            f"distance {dist:.0f}m. Own vessel is give-way vessel."
                        ),
                    ))
                elif 170 < bearing < 350 and bearing > 10:
                    # Other vessel on our port side — we are stand-on
                    urgency = UrgencyLevel.CAUTIONARY if dist < 500 else UrgencyLevel.INFORMATIONAL
                    results.append(COLREGsResult(
                        applicable_rules=[rule] if rule else [],
                        recommended_action=(
                            f"Maintain course and speed; '{other.name}' should give way. "
                            f"Monitor for compliance and be prepared to take avoiding action."
                        ),
                        urgency=urgency,
                        rule_references=["Rule 15", "Rule 17"],
                        give_way_vessel=other.name,
                        stand_on_vessel=situation.own_vessel.name or "own vessel",
                        reasoning=(
                            f"Crossing situation: vessel '{other.name}' "
                            f"at bearing {bearing:.1f}° (port side), "
                            f"distance {dist:.0f}m. Own vessel is stand-on vessel."
                        ),
                    ))

        if results:
            return max(results, key=lambda r: (
                0 if r.urgency == UrgencyLevel.IMMEDIATE
                else 1 if r.urgency == UrgencyLevel.URGENT
                else 2 if r.urgency == UrgencyLevel.CAUTIONARY
                else 3
            ))

        return COLREGsResult(
            applicable_rules=[],
            recommended_action="No crossing situation detected",
            urgency=UrgencyLevel.INFORMATIONAL,
            rule_references=[],
        )

    def check_overtaking(self, situation: VesselSituation) -> COLREGsResult:
        """Check for overtaking situations (Rule 13).

        Any vessel overtaking any other shall keep out of the way.

        Args:
            situation: Current navigation situation

        Returns:
            COLREGsResult with overtaking assessment
        """
        rule = self._get_rule("Rule 13")
        results = []

        for other in situation.other_vessels:
            bearing = self._bearing_to(situation.own_vessel, other)
            dist = self._distance_to(situation.own_vessel, other)

            # Overtaking: approaching from behind (within ±67.5° of stern, i.e., 112.5°-247.5°)
            if dist < 1000 and (112.5 <= bearing <= 247.5):
                # Check if we're actually closing in
                if other.speed < situation.own_vessel.speed:
                    urgency = UrgencyLevel.URGENT if dist < 200 else UrgencyLevel.CAUTIONARY
                    results.append(COLREGsResult(
                        applicable_rules=[rule] if rule else [],
                        recommended_action=(
                            f"Keep out of the way of '{other.name}' until well clear. "
                            f"Pass on the port side if safe."
                        ),
                        urgency=urgency,
                        rule_references=["Rule 13"],
                        give_way_vessel=situation.own_vessel.name or "own vessel",
                        stand_on_vessel=other.name,
                        reasoning=(
                            f"Overtaking situation: vessel '{other.name}' "
                            f"at bearing {bearing:.1f}° (within overtaking arc), "
                            f"distance {dist:.0f}m. Own vessel is overtaking vessel "
                            f"and must keep clear."
                        ),
                    ))

            # Also check if another vessel is overtaking us
            other_bearing = self._bearing_to(other, situation.own_vessel)
            if dist < 1000 and (112.5 <= other_bearing <= 247.5):
                if situation.own_vessel.speed < other.speed:
                    urgency = UrgencyLevel.CAUTIONARY
                    results.append(COLREGsResult(
                        applicable_rules=[rule] if rule else [],
                        recommended_action=(
                            f"Maintain course and speed; '{other.name}' is overtaking "
                            f"and must keep clear."
                        ),
                        urgency=urgency,
                        rule_references=["Rule 13", "Rule 17"],
                        give_way_vessel=other.name,
                        stand_on_vessel=situation.own_vessel.name or "own vessel",
                        reasoning=(
                            f"Being overtaken by '{other.name}' "
                            f"at relative bearing {other_bearing:.1f}°, "
                            f"distance {dist:.0f}m. Maintain course and speed."
                        ),
                    ))

        if results:
            return max(results, key=lambda r: (
                0 if r.urgency == UrgencyLevel.IMMEDIATE
                else 1 if r.urgency == UrgencyLevel.URGENT
                else 2
            ))

        return COLREGsResult(
            applicable_rules=[],
            recommended_action="No overtaking situation detected",
            urgency=UrgencyLevel.INFORMATIONAL,
            rule_references=[],
        )

    def check_stand_on_give_way(self, situation: VesselSituation) -> COLREGsResult:
        """Check stand-on/give-way determination for a situation.

        Comprehensive check combining Rules 12-19.

        Args:
            situation: Current navigation situation

        Returns:
            COLREGsResult with stand-on/give-way determination
        """
        applicable = []
        all_actions = []
        max_urgency = UrgencyLevel.INFORMATIONAL
        all_reasoning = []

        # Check head-on first (highest priority)
        head_on = self.check_head_on_situation(situation)
        if head_on.applicable_rules:
            applicable.extend(head_on.applicable_rules)
            all_actions.append(head_on.recommended_action)
            max_urgency = self._max_urgency(max_urgency, head_on.urgency)
            all_reasoning.append(head_on.reasoning)

        # Check crossing
        crossing = self.check_crossing_situation(situation)
        if crossing.applicable_rules:
            applicable.extend(crossing.applicable_rules)
            all_actions.append(crossing.recommended_action)
            max_urgency = self._max_urgency(max_urgency, crossing.urgency)
            all_reasoning.append(crossing.reasoning)

        # Check overtaking
        overtaking = self.check_overtaking(situation)
        if overtaking.applicable_rules:
            applicable.extend(overtaking.applicable_rules)
            all_actions.append(overtaking.recommended_action)
            max_urgency = self._max_urgency(max_urgency, overtaking.urgency)
            all_reasoning.append(overtaking.reasoning)

        # Determine give-way and stand-on vessels
        give_way = crossing.give_way_vessel or head_on.give_way_vessel or ""
        stand_on = crossing.stand_on_vessel or head_on.stand_on_vessel or ""

        return COLREGsResult(
            applicable_rules=applicable,
            recommended_action="; ".join(all_actions) if all_actions else "No collision risk detected",
            urgency=max_urgency,
            rule_references=list(set(
                head_on.rule_references + crossing.rule_references + overtaking.rule_references
            )),
            give_way_vessel=give_way,
            stand_on_vessel=stand_on,
            reasoning=" | ".join(all_reasoning) if all_reasoning else "All clear",
        )

    def _max_urgency(self, a: UrgencyLevel, b: UrgencyLevel) -> UrgencyLevel:
        """Return the more urgent of two urgency levels."""
        order = {
            UrgencyLevel.IMMEDIATE: 0,
            UrgencyLevel.URGENT: 1,
            UrgencyLevel.CAUTIONARY: 2,
            UrgencyLevel.INFORMATIONAL: 3,
        }
        return a if order.get(a, 99) <= order.get(b, 99) else b

    def check_navigation_lights(self, situation: VesselSituation) -> COLREGsResult:
        """Check navigation lights compliance (Rules 20-32).

        Args:
            situation: Current navigation situation

        Returns:
            COLREGsResult with lights compliance assessment
        """
        rule = self._get_rule("Rule 20")
        findings = []

        # Check during nighttime or restricted visibility
        is_night = situation.time_of_day == "night"
        is_restricted = situation.visibility in (
            VisibilityCondition.RESTRICTED, VisibilityCondition.POOR
        )

        if not is_night and not is_restricted:
            return COLREGsResult(
                applicable_rules=[],
                recommended_action="Navigation lights not required in current conditions",
                urgency=UrgencyLevel.INFORMATIONAL,
                rule_references=[],
            )

        applicable_rules = [rule] if rule else []

        # Check own vessel lights
        if not situation.own_vessel.has_night_lights:
            findings.append(f"Own vessel '{situation.own_vessel.name}' missing navigation lights")
            return COLREGsResult(
                applicable_rules=applicable_rules,
                recommended_action="ACTIVATE navigation lights immediately",
                urgency=UrgencyLevel.IMMEDIATE,
                rule_references=["Rule 20"],
                reasoning="Navigation lights required but not active on own vessel",
            )

        # Check other vessels
        for other in situation.other_vessels:
            dist = self._distance_to(situation.own_vessel, other)
            if dist < 5000:
                if not other.has_night_lights:
                    findings.append(
                        f"Vessel '{other.name}' at {dist:.0f}m not displaying navigation lights"
                    )

        if findings:
            return COLREGsResult(
                applicable_rules=applicable_rules,
                recommended_action="Exercise extreme caution; vessels without lights detected in area",
                urgency=UrgencyLevel.URGENT,
                rule_references=["Rule 20"],
                reasoning="; ".join(findings),
            )

        return COLREGsResult(
            applicable_rules=applicable_rules,
            recommended_action="All vessels displaying proper navigation lights",
            urgency=UrgencyLevel.INFORMATIONAL,
            rule_references=["Rule 20"],
            reasoning="Navigation lights compliance verified for all vessels in range",
        )

    def check_sound_signals(self, situation: VesselSituation) -> COLREGsResult:
        """Check sound signals compliance (Rules 34-37).

        Args:
            situation: Current navigation situation

        Returns:
            COLREGsResult with sound signals assessment
        """
        rule_34 = self._get_rule("Rule 34")
        rule_35 = self._get_rule("Rule 35")

        is_restricted = situation.visibility in (
            VisibilityCondition.RESTRICTED, VisibilityCondition.POOR
        )

        applicable_rules = []
        recommendations = []
        reasoning_parts = []

        # Rule 35: Restricted visibility sound signals
        if is_restricted:
            applicable_rules.append(rule_35)
            if not situation.own_vessel.has_sound_signals and situation.own_vessel.speed > 0:
                recommendations.append(
                    "Sound prolonged blast at intervals not exceeding 2 minutes (Rule 35)"
                )
                reasoning_parts.append(
                    "Sound signals required in restricted visibility but not active"
                )

        # Rule 34: Manoeuvring signals when other vessels present
        for other in situation.other_vessels:
            dist = self._distance_to(situation.own_vessel, other)
            if dist < 1000 and self._is_approaching(situation.own_vessel, other):
                if not is_restricted:
                    applicable_rules.append(rule_34)
                    recommendations.append(
                        "Sound appropriate manoeuvring signals when altering course (Rule 34)"
                    )
                    reasoning_parts.append(
                        f"Vessel '{other.name}' within signalling range ({dist:.0f}m)"
                    )
                    break

        if not recommendations:
            if is_restricted:
                return COLREGsResult(
                    applicable_rules=[r for r in applicable_rules if r],
                    recommended_action="Continue sounding appropriate fog signals",
                    urgency=UrgencyLevel.CAUTIONARY,
                    rule_references=["Rule 35"],
                    reasoning="Sound signals compliant in restricted visibility",
                )
            return COLREGsResult(
                applicable_rules=[],
                recommended_action="No sound signals required in current conditions",
                urgency=UrgencyLevel.INFORMATIONAL,
                rule_references=[],
            )

        return COLREGsResult(
            applicable_rules=[r for r in applicable_rules if r],
            recommended_action="; ".join(recommendations),
            urgency=UrgencyLevel.URGENT if is_restricted else UrgencyLevel.CAUTIONARY,
            rule_references=["Rule 34", "Rule 35"],
            reasoning="; ".join(reasoning_parts),
        )

    def check_conduct_in_restricted_visibility(self, situation: VesselSituation) -> COLREGsResult:
        """Check conduct in restricted visibility (Rule 19).

        Args:
            situation: Current navigation situation

        Returns:
            COLREGsResult with restricted visibility assessment
        """
        rule = self._get_rule("Rule 19")

        if situation.visibility not in (
            VisibilityCondition.RESTRICTED, VisibilityCondition.POOR
        ):
            return COLREGsResult(
                applicable_rules=[],
                recommended_action="Rule 19 not applicable in current visibility",
                urgency=UrgencyLevel.INFORMATIONAL,
                rule_references=[],
            )

        recommendations = []
        reasoning_parts = []

        # Check safe speed
        if situation.own_vessel.speed > 5.0:
            recommendations.append(
                f"Reduce speed from {situation.own_vessel.speed:.1f} knots to safe speed for restricted visibility"
            )
            reasoning_parts.append(
                f"Speed {situation.own_vessel.speed:.1f} knots may be unsafe for restricted visibility"
            )

        # Check for approaching vessels
        for other in situation.other_vessels:
            dist = self._distance_to(situation.own_vessel, other)
            if dist < 500:
                recommendations.append(
                    f"Take avoiding action for vessel '{other.name}' at {dist:.0f}m — "
                    f"avoid altering course to port for a vessel forward of the beam"
                )
                reasoning_parts.append(
                    f"Close approach detected: '{other.name}' at {dist:.0f}m in restricted visibility"
                )

        if not recommendations:
            return COLREGsResult(
                applicable_rules=[rule] if rule else [],
                recommended_action=(
                    "Maintain current safe speed and navigation practices; "
                    "engines ready for immediate manoeuvre"
                ),
                urgency=UrgencyLevel.CAUTIONARY,
                rule_references=["Rule 19"],
                reasoning="Proper conduct observed in restricted visibility",
            )

        return COLREGsResult(
            applicable_rules=[rule] if rule else [],
            recommended_action="; ".join(recommendations),
            urgency=UrgencyLevel.URGENT,
            rule_references=["Rule 19"],
            reasoning="; ".join(reasoning_parts),
        )
