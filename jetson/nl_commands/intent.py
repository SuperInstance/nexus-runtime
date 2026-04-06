"""
Intent Recognition and Classification for NEXUS Marine Robotics Platform.

Implements rule-based intent recognition with slot extraction, confidence
scoring, disambiguation, and training via example patterns.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes and enums
# ---------------------------------------------------------------------------

class IntentType(Enum):
    """High-level intent categories for marine robot commands."""
    NAVIGATE = "navigate"
    STATION_KEEP = "station_keep"
    PATROL = "patrol"
    SURVEY = "survey"
    EMERGENCY_STOP = "emergency_stop"
    RETURN_HOME = "return_home"
    SET_SPEED = "set_speed"
    SET_HEADING = "set_heading"
    QUERY_STATUS = "query_status"
    CONFIGURE = "configure"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """A recognized user intent."""
    type: IntentType
    slots: dict[str, object] = field(default_factory=dict)
    confidence: float = 0.0
    raw_text: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            import time as _time
            self.timestamp = _time.time()


# ---------------------------------------------------------------------------
# Intent patterns (keyword-based)
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: dict[IntentType, list[str]] = {
    IntentType.NAVIGATE: [
        r"\bgo\b", r"\bmove\b", r"\bnavigate\b", r"\btravel\b", r"\bhead\b",
        r"\bsteer\b", r"\bsail\b", r"\bcruise\b", r"\bproceed\b", r"\bfollow\b",
        r"\btrack\b", r"\bgoto\b", r"\bgo to\b", r"\bgo straight\b",
    ],
    IntentType.STATION_KEEP: [
        r"\bhold\b", r"\bstay\b", r"\bkeep\b", r"\bremain\b", r"\bmaintain position\b",
        r"\bstation keep\b", r"\bhover\b", r"\bloiter\b", r"\bstop here\b",
    ],
    IntentType.PATROL: [
        r"\bpatrol\b", r"\bguard\b", r"\bprotect\b", r"\bdefend\b",
        r"\bmonitor area\b", r"\bsecurity sweep\b", r"\bcirculate\b", r"\bcircuit\b",
    ],
    IntentType.SURVEY: [
        r"\bsurvey\b", r"\bscan\b", r"\bmap\b", r"\bsearch\b", r"\bexplore\b",
        r"\binspect\b", r"\bobserve\b", r"\bmonitor\b", r"\brecon\b",
    ],
    IntentType.EMERGENCY_STOP: [
        r"\bemergency stop\b", r"\bastop\b", r"\bstop\b", r"\babort\b", r"\bcancel\b",
        r"\bhalt\b", r"\bemergency\b", r"\ball stop\b", r"\bdead stop\b",
        r"\bkill\b", r"\bshutdown\b",
    ],
    IntentType.RETURN_HOME: [
        r"\breturn\b", r"\bgo home\b", r"\bcome back\b", r"\brtb\b",
        r"\bhome\b", r"\bdock\b", r"\brecall\b", r"\bhead home\b",
    ],
    IntentType.SET_SPEED: [
        r"\bset speed\b", r"\bspeed\b", r"\bgo faster\b", r"\bslow down\b",
        r"\bincrease speed\b", r"\bdecrease speed\b", r"\bfull speed\b",
        r"\bhalf speed\b", r"\baccelerate\b", r"\bdecelerate\b",
    ],
    IntentType.SET_HEADING: [
        r"\bheading\b", r"\bbearing\b", r"\bcourse\b", r"\bturn\b",
        r"\bsteer to\b", r"\bface\b", r"\bpoint\b", r"\brotate\b",
    ],
    IntentType.QUERY_STATUS: [
        r"\bstatus\b", r"\breport\b", r"\btell\b", r"\bshow\b",
        r"\bquery\b", r"\bcheck\b", r"\bwhat\b", r"\bhow\b",
        r"\bwhere\b", r"\bcurrent\b", r"\bposition\b", r"\bbattery\b",
    ],
    IntentType.CONFIGURE: [
        r"\bconfigure\b", r"\bset\b", r"\benable\b", r"\bdisable\b",
        r"\bturn on\b", r"\bturn off\b", r"\bactivate\b", r"\bdeactivate\b",
        r"\bcalibrate\b", r"\bmode\b", r"\bsettings?\b", r"\badjust\b",
    ],
}

# Slot extraction patterns per intent
_SLOT_PATTERNS: dict[IntentType, dict[str, list[str]]] = {
    IntentType.NAVIGATE: {
        "destination": [r"\b(?:to|toward|towards)\s+(.+?)(?:\s*\.|$)", r"\bgo\s+(.+?)(?:\s*\.|$)"],
        "speed": [r"(\d+(?:\.\d+)?)\s*(?:knot|knots|kn)\b"],
        "waypoint": [r"\bwaypoint\s+(\w+)", r"\bwp\s+(\w+)"],
    },
    IntentType.SET_SPEED: {
        "speed": [r"(\d+(?:\.\d+)?)\s*(?:knot|knots|kn|m/s|kph|mph)\b"],
        "level": [r"\b(full|half|quarter|minimum|maximum|crawl|cruise)\b"],
    },
    IntentType.SET_HEADING: {
        "heading_degrees": [r"(\d{1,3}(?:\.\d+)?)\s*(?:degrees?|°)?\b"],
        "direction": [r"\b(north|south|east|west|northeast|northwest|southeast|southwest|north-northwest|north-northeast|south-southeast|south-southwest)\b"],
    },
    IntentType.SURVEY: {
        "area": [r"\b(?:survey|scan|map|search)\s+(?:the\s+)?(\w+(?:\s+\w+)?)"],
        "pattern": [r"\b(lawnmower|zigzag|spiral|radial|grid|linear)\s+(?:pattern|sweep)"],
    },
    IntentType.PATROL: {
        "zone": [r"\bpatrol\s+(?:the\s+)?(\w+(?:\s+\w+)?)", r"\bguard\s+(?:the\s+)?(\w+(?:\s+\w+)?)"],
        "pattern": [r"\b(circular|perimeter|figure-eight|random)\b"],
    },
    IntentType.STATION_KEEP: {
        "duration": [r"(\d+)\s*(?:seconds?|minutes?|hours?|mins?|hrs?|secs?)\b"],
        "radius": [r"(\d+(?:\.\d+)?)\s*(?:meter|m|feet|ft|nm|miles?)\s+(?:radius|tolerance)"],
    },
    IntentType.RETURN_HOME: {
        "speed": [r"(\d+(?:\.\d+)?)\s*(?:knot|knots|kn)\b"],
    },
    IntentType.CONFIGURE: {
        "parameter": [r"\b(?:set|enable|disable|activate|deactivate|configure|adjust)\s+(?:the\s+)?(\w+(?:\s+\w+)?)"],
        "value": [r"\b(?:to|=)\s+(\w+(?:\s+\w+)?)"],
    },
    IntentType.QUERY_STATUS: {
        "subject": [r"\b(?:status|report|check)\s+(?:of\s+|for\s+)?(\w+(?:\s+\w+)?)"],
    },
    IntentType.EMERGENCY_STOP: {},
}

# Ambiguity resolution priority
_INTENT_PRIORITY = [
    IntentType.EMERGENCY_STOP,
    IntentType.NAVIGATE,
    IntentType.STATION_KEEP,
    IntentType.PATROL,
    IntentType.SURVEY,
    IntentType.RETURN_HOME,
    IntentType.SET_SPEED,
    IntentType.SET_HEADING,
    IntentType.QUERY_STATUS,
    IntentType.CONFIGURE,
    IntentType.UNKNOWN,
]


# ---------------------------------------------------------------------------
# IntentRecognizer
# ---------------------------------------------------------------------------

class IntentRecognizer:
    """Rule-based intent recognition for marine NL commands.

    Uses keyword/pattern matching to classify user input into intent types,
    extract relevant slots, and compute confidence scores.
    """

    def __init__(self) -> None:
        self._patterns: dict[IntentType, list[re.Pattern]] = {
            itype: [re.compile(p, re.IGNORECASE) for p in pats]
            for itype, pats in _INTENT_PATTERNS.items()
        }
        self._slot_patterns: dict[IntentType, dict[str, list[re.Pattern]]] = {
            itype: {
                sname: [re.compile(p, re.IGNORECASE) for p in spats]
                for sname, spats in slots.items()
            }
            for itype, slots in _SLOT_PATTERNS.items()
        }
        self._custom_examples: list[tuple[str, IntentType]] = []

    # -- public API --------------------------------------------------------

    def recognize(self, text: str, context: Optional[dict] = None) -> Intent:
        """Recognize the intent behind *text*.

        *context* is an optional dict that may influence recognition (e.g.
        current operating mode, last intent, etc.).
        """
        text_norm = text.strip()
        if not text_norm:
            return self._make_intent(IntentType.UNKNOWN, 0.0, text)

        candidates = self._score_all_intents(text_norm, context)

        if len(candidates) > 1:
            best = self.disambiguate(candidates)
        elif len(candidates) == 1:
            best = candidates[0]
        else:
            best = self._make_intent(IntentType.UNKNOWN, 0.2, text)

        # Extract slots for the recognized intent
        best.slots = self.extract_slots(text_norm, best.type)

        return best

    def train(self, examples: list[tuple[str, IntentType]]) -> float:
        """Add custom training examples and return estimated accuracy.

        *examples* is a list of (text, intent_type) tuples. The method
        adds them to an internal pattern set and returns a self-assessed
        accuracy score.
        """
        correct = 0
        for text, expected in examples:
            self._custom_examples.append((text, expected))
            result = self.recognize(text)
            if result.type == expected:
                correct += 1

        accuracy = correct / len(examples) if examples else 0.0
        return round(accuracy, 4)

    def compute_confidence(self, text: str, intent_type: IntentType) -> float:
        """Compute a confidence score [0.0, 1.0] for *text* matching *intent_type*."""
        patterns = self._patterns.get(intent_type, [])
        if not patterns:
            return 0.0
        text_lower = text.lower()
        hits = sum(1 for p in patterns if p.search(text_lower))
        if hits == 0:
            return 0.0
        return min(1.0, hits * 0.3)

    def disambiguate(self, intents: list[Intent]) -> Intent:
        """Select the best intent from a list of candidates."""
        if not intents:
            return self._make_intent(IntentType.UNKNOWN, 0.0, "")

        if len(intents) == 1:
            return intents[0]

        # Sort by priority first, then confidence
        def sort_key(intent: Intent) -> tuple[int, float]:
            try:
                priority = _INTENT_PRIORITY.index(intent.type)
            except ValueError:
                priority = len(_INTENT_PRIORITY)
            return (priority, -intent.confidence)

        sorted_intents = sorted(intents, key=sort_key)
        return sorted_intents[0]

    def extract_slots(self, text: str, intent_type: IntentType) -> dict[str, object]:
        """Extract parameter slots from *text* for the given *intent_type*."""
        slots: dict[str, object] = {}
        slot_defs = self._slot_patterns.get(intent_type, {})
        for slot_name, patterns in slot_defs.items():
            for pat in patterns:
                m = pat.search(text)
                if m:
                    val = m.group(1)
                    # Try numeric conversion
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    slots[slot_name] = val
                    break  # first match wins
        return slots

    def get_supported_intents(self) -> list[IntentType]:
        """Return all supported intent types."""
        return list(IntentType)

    # -- internal helpers ---------------------------------------------------

    def _score_all_intents(self, text: str, context: Optional[dict]) -> list[Intent]:
        """Score *text* against all intent patterns and return candidates above threshold."""
        candidates: list[Intent] = []
        for itype, patterns in self._patterns.items():
            conf = self.compute_confidence(text, itype)
            if conf > 0.0:
                intent = self._make_intent(itype, conf, text)
                # Boost from context
                if context:
                    conf = self._apply_context_boost(intent, context)
                    intent.confidence = min(1.0, conf)
                candidates.append(intent)

        # Check custom examples
        for ex_text, ex_type in self._custom_examples:
            if self._text_similarity(text, ex_text) > 0.7:
                conf = self._text_similarity(text, ex_text)
                candidates.append(self._make_intent(ex_type, conf, text))

        # Filter: keep only candidates with confidence > 0.1
        candidates = [c for c in candidates if c.confidence > 0.1]
        return candidates

    @staticmethod
    def _make_intent(itype: IntentType, conf: float, text: str) -> Intent:
        return Intent(
            type=itype,
            slots={},
            confidence=round(conf, 4),
            raw_text=text,
            timestamp=time.time(),
        )

    @staticmethod
    def _apply_context_boost(intent: Intent, context: dict) -> float:
        """Boost confidence based on dialogue context."""
        boost = 0.0
        last_intent = context.get("last_intent")
        if last_intent and isinstance(last_intent, IntentType) and last_intent == intent.type:
            boost = 0.1
        mode = context.get("mode", "")
        if mode == "emergency" and intent.type == IntentType.EMERGENCY_STOP:
            boost += 0.15
        if mode == "survey" and intent.type in (IntentType.SURVEY, IntentType.NAVIGATE):
            boost += 0.05
        return intent.confidence + boost

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
