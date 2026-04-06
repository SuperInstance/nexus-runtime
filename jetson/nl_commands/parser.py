"""
Natural Language Command Parser for NEXUS Marine Robotics Platform.

Implements tokenization, parsing, entity extraction, and command normalization
using pure Python (regex + string operations). No external NLP dependencies.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Token:
    """A single token in the input text."""
    text: str
    pos_tag: str        # rough POS: NOUN, VERB, ADJ, ADV, PREP, DET, NUM, COORD, PUNCT, UNKNOWN
    lemma: str          # normalized / lower-cased base form
    start: int
    end: int


@dataclass
class Entity:
    """An extracted entity (location, speed, heading, distance, etc.)."""
    entity_type: str    # e.g. "coordinate", "speed", "heading", "distance", "duration", "waypoint"
    value: object       # extracted value (number, string, tuple, etc.)
    raw_text: str       # original span
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class ParseTree:
    """Result of parsing an NL command string."""
    tokens: list[Token]
    root: str                           # root verb / command head
    entities: list[Entity]
    confidence: float


# ---------------------------------------------------------------------------
# Simple POS tagger (rule-based)
# ---------------------------------------------------------------------------

# Determiners
_DETERMINERS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "their",
    "some", "any", "each", "every", "all", "both", "few", "many", "much", "several",
    "no", "none", "its", "his", "her",
})

# Common prepositions
_PREPOSITIONS = frozenset({
    "to", "from", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "of", "off", "over", "under", "up", "down", "out", "around", "toward", "towards",
    "near", "past", "along", "across", "behind", "beside", "beyond", "without",
})

# Coordinating conjunctions
_COORDS = frozenset({"and", "or", "but", "nor", "so", "yet", "for"})

# Common verbs for marine commands
_VERBS = frozenset({
    "go", "move", "navigate", "travel", "head", "steer", "sail", "cruise",
    "stop", "halt", "pause", "return", "come", "back", "follow", "track",
    "set", "adjust", "change", "increase", "decrease", "reduce", "raise", "lower",
    "start", "begin", "launch", "resume", "continue",
    "scan", "survey", "map", "search", "look", "monitor", "watch",
    "patrol", "guard", "protect", "defend", "secure",
    "hold", "keep", "stay", "remain", "maintain",
    "report", "tell", "show", "display", "give", "query", "check", "get",
    "configure", "enable", "disable", "turn", "activate", "deactivate",
    "abort", "cancel", "emergency", "evacuate", "rescue",
    "dive", "surface", "descend", "ascend",
    "anchor", "dock", "moor", "deploy", "retrieve", "drop", "pick",
    "avoid", "circumnavigate", "orbit", "circle",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "have", "has", "had", "can", "could", "will", "would", "should", "may", "might", "must",
    "need", "want", "please", "let",
})

# Common nouns for marine domain
_NOUNS = frozenset({
    "waypoint", "coordinates", "location", "position", "destination", "target",
    "speed", "heading", "course", "bearing", "direction", "distance", "depth",
    "engine", "motor", "thruster", "propeller", "rudder", "sonar", "camera",
    "sensor", "battery", "power", "fuel", "range", "mode",
    "mission", "task", "operation", "zone", "area", "region", "perimeter",
    "vessel", "boat", "ship", "submarine", "drone", "robot", "surface", "vehicle",
    "port", "starboard", "bow", "stern", "north", "south", "east", "west",
    "nautical", "mile", "miles", "knot", "knots", "meter", "meters", "feet", "foot",
    "degree", "degrees", "minute", "minutes", "second", "seconds", "hour", "hours",
    "status", "report", "data", "information", "log", "map", "chart",
    "obstacle", "hazard", "shore", "coast", "harbor", "dock", "bay", "channel",
    "latitude", "longitude", "altitude", "frequency", "pattern", "formation",
    "crew", "system", "controller", "autopilot", "gps", "lidar", "radar",
})

# Adjectives
_ADJS = frozenset({
    "north", "south", "east", "west", "northern", "southern", "eastern", "western",
    "fast", "slow", "quick", "quickly", "safe", "unsafe", "dangerous",
    "high", "low", "maximum", "minimum", "full", "half", "quarter",
    "left", "right", "forward", "backward", "straight", "lateral",
    "autonomous", "manual", "automatic", "remote", "local",
    "active", "inactive", "enabled", "disabled", "on", "off",
    "urgent", "critical", "normal", "routine", "primary", "secondary",
    "circular", "linear", "zigzag", "spiral", "radial",
    "shallow", "deep", "narrow", "wide", "open", "close", "clear",
    "new", "old", "current", "previous", "next", "last",
})

# Adverbs
_ADVS = frozenset({
    "quickly", "slowly", "safely", "carefully", "immediately", "now",
    "then", "there", "here", "always", "never", "often", "sometimes",
    "very", "quite", "too", "also", "just", "only", "still", "already",
    "please", "gently", "gradually", "steadily", "directly", "precisely",
    "ahead", "astern", "aboard", "ashore", "afloat", "aground",
})


def _simple_pos(word: str) -> str:
    """Assign a rough POS tag to *word* based on dictionaries."""
    w = word.lower()
    if re.fullmatch(r"-?[+]?\d+(\.\d+)?", w):
        return "NUM"
    if w in _DETERMINERS:
        return "DET"
    if w in _COORDS:
        return "COORD"
    if w in _PREPOSITIONS:
        return "PREP"
    if re.fullmatch(r"[.,;:!?]", w):
        return "PUNCT"
    if w in _VERBS:
        return "VERB"
    if w in _NOUNS:
        return "NOUN"
    if w in _ADJS:
        return "ADJ"
    if w in _ADVS:
        return "ADV"
    # heuristics for words ending in -ly
    if w.endswith("ly") and len(w) > 3:
        return "ADV"
    if w.endswith("ing"):
        return "VERB"
    if w.endswith("tion") or w.endswith("ment") or w.endswith("ness") or w.endswith("ity"):
        return "NOUN"
    if w.endswith("ous") or w.endswith("ive") or w.endswith("able") or w.endswith("ful"):
        return "ADJ"
    return "UNKNOWN"


def _simple_lemma(word: str) -> str:
    """Very rough lemmatization."""
    w = word.lower().strip()
    # irregular forms
    _irregular = {
        "is": "be", "are": "be", "was": "be", "were": "be", "am": "be",
        "been": "be", "being": "be",
        "has": "have", "had": "have", "having": "have",
        "does": "do", "did": "do", "doing": "do",
        "went": "go", "gone": "go", "going": "go",
        "moved": "move", "moving": "move",
        "stopped": "stop", "stopping": "stop",
        "navigated": "navigate", "navigating": "navigate",
        "steered": "steer", "steering": "steer",
        "setted": "set", "setting": "set",
        "started": "start", "starting": "start",
        "scanned": "scan", "scanning": "scan",
        "patrolled": "patrol", "patrolling": "patrol",
        "held": "hold", "holding": "hold",
        "reported": "report", "reporting": "report",
        "configured": "configure", "configuring": "configure",
        "increased": "increase", "increasing": "increase",
        "decreased": "decrease", "decreasing": "decrease",
        "returned": "return", "returning": "return",
        "followed": "follow", "following": "follow",
        "dived": "dive", "diving": "dive",
        "surfaced": "surface", "surfacing": "surface",
        "anchored": "anchor", "anchoring": "anchor",
        "moored": "moor", "mooring": "moor",
        "deployed": "deploy", "deploying": "deploy",
        "circled": "circle", "circling": "circle",
        "orbits": "orbit", "orbiting": "orbit",
        "knots": "knot", "miles": "mile", "meters": "meter",
        "feet": "foot", "degrees": "degree",
        "coordinates": "coordinate",
        "waypoints": "waypoint",
        "engines": "engine", "motors": "motor", "thrusters": "thruster",
        "sensors": "sensor", "batteries": "battery",
        "missions": "mission", "tasks": "task", "operations": "operation",
        "obstacles": "obstacle", "hazards": "hazard",
        "minutes": "minute", "seconds": "second", "hours": "hour",
    }
    if w in _irregular:
        return _irregular[w]
    # simple suffix stripping
    if w.endswith("ing") and len(w) > 5:
        return w[:-3] if w[-4] == w[-5] else w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2] if w[-3] == w[-4] else w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("ly") and len(w) > 4:
        return w[:-2]
    return w


# ---------------------------------------------------------------------------
# NLParser
# ---------------------------------------------------------------------------

class NLParser:
    """Natural language command parser.

    All parsing is implemented with regex and simple rule matching.
    No external NLP libraries are used.
    """

    # Patterns
    _COORD_PATTERN = re.compile(
        r"(?:(\d{1,3})\s*[°d]\s*(\d{1,2}(?:\.\d+)?)\s*['m]\s*(\d{1,2}(?:\.\d+)?)\s*[\"s]?\s*([NSEWnsew]))"
        r"|"
        r"(?:(-?\d{1,3}(?:\.\d+))\s*[°]?\s*([NSEWnsew])\s*[,;\s]+(-?\d{1,3}(?:\.\d+))\s*[°]?\s*([NSEWnsew]))"
    )
    _DECIMAL_COORD_PATTERN = re.compile(
        r"(-?\d{1,3}(?:\.\d+))\s*[,;\s]+\s*(-?\d{1,3}(?:\.\d+))"
    )
    _NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")
    _DURATION_PATTERN = re.compile(
        r"(\d+)\s*(second|seconds|sec|minute|minutes|min|hour|hours|hr|h)\b",
        re.IGNORECASE,
    )
    _SPEED_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*(knot|knots|kn|kph|km/h|mph|m/s|meter(?:s)?/s)\b",
        re.IGNORECASE,
    )
    _HEADING_PATTERN = re.compile(
        r"(?:heading|bearing|course)\s+(?:of\s+|to\s+)?(\d{1,3}(?:\.\d+)?)\s*(?:degrees?|°)?\b",
        re.IGNORECASE,
    )
    _DISTANCE_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*(nautical\s*mile|nautical\s*miles|nm|miles?|mi|kilometer|kilometers|km|meter|meters|m|feet|ft|yard|yards|yd)\b",
        re.IGNORECASE,
    )

    _DURATION_UNITS = {
        "second": 1, "seconds": 1, "sec": 1,
        "minute": 60, "minutes": 60, "min": 60,
        "hour": 3600, "hours": 3600, "hr": 3600, "h": 3600,
    }

    def __init__(self) -> None:
        self._entity_cache: dict[str, list[Entity]] = {}

    # -- public API --------------------------------------------------------

    def tokenize(self, text: str) -> list[Token]:
        """Split *text* into tokens with POS tags and lemmas."""
        tokens: list[Token] = []
        # Use regex to split on whitespace and punctuation boundaries while keeping punctuation
        parts = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[.,;:!?]|[-+]", text)
        cursor = 0
        for part in parts:
            idx = text.find(part, cursor)
            if idx == -1:
                idx = cursor
            tokens.append(Token(
                text=part,
                pos_tag=_simple_pos(part),
                lemma=_simple_lemma(part),
                start=idx,
                end=idx + len(part),
            ))
            cursor = idx + len(part)
        return tokens

    def parse(self, text: str) -> ParseTree:
        """Full parse: tokenize + extract root verb + entities."""
        tokens = self.tokenize(text)
        root = self._find_root_verb(tokens)
        entities = self.extract_entities(text)
        confidence = self._compute_parse_confidence(tokens, entities)
        return ParseTree(
            tokens=tokens,
            root=root,
            entities=entities,
            confidence=confidence,
        )

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract structured entities from *text*."""
        entities: list[Entity] = []
        seen_spans: set[tuple[int, int]] = set()

        def _add(etype: str, value: object, raw: str, start: int, end: int, conf: float = 1.0):
            span = (start, end)
            if span not in seen_spans:
                seen_spans.add(span)
                entities.append(Entity(entity_type=etype, value=value, raw_text=raw, start=start, end=end, confidence=conf))

        # Coordinates
        for coord in self.extract_coordinates(text):
            if isinstance(coord, tuple) and len(coord) == 2:
                # find the raw text
                m = self._COORD_PATTERN.search(text)
                if m:
                    _add("coordinate", coord, m.group(0), m.start(), m.end())
                else:
                    m2 = self._DECIMAL_COORD_PATTERN.search(text)
                    if m2:
                        _add("coordinate", coord, m2.group(0), m2.start(), m2.end())

        # Speeds
        for m in self._SPEED_PATTERN.finditer(text):
            val = float(m.group(1))
            _add("speed", val, m.group(0), m.start(), m.end())

        # Headings
        for m in self._HEADING_PATTERN.finditer(text):
            val = float(m.group(1))
            _add("heading", val, m.group(0), m.start(), m.end())

        # Distances
        for m in self._DISTANCE_PATTERN.finditer(text):
            val = float(m.group(1))
            _add("distance", val, m.group(0), m.start(), m.end())

        # Durations
        for m in self._DURATION_PATTERN.finditer(text):
            unit = self._DURATION_UNITS.get(m.group(2).lower(), 1)
            val = int(m.group(1)) * unit
            _add("duration", val, m.group(0), m.start(), m.end())

        # Plain numbers (not already captured)
        for m in self._NUMBER_PATTERN.finditer(text):
            span = (m.start(), m.end())
            if span not in seen_spans:
                _add("number", float(m.group(0)), m.group(0), m.start(), m.end())

        # Sort by start position
        entities.sort(key=lambda e: e.start)
        return entities

    def extract_numbers(self, text: str) -> list[float]:
        """Extract all numeric values from *text*."""
        return [float(m.group(0)) for m in self._NUMBER_PATTERN.finditer(text)]

    def extract_coordinates(self, text: str) -> list[tuple[float, float]]:
        """Extract (lat, lon) coordinate pairs from *text*."""
        coords: list[tuple[float, float]] = []

        # DMS format: 37°45'23"N 122°27'00"W
        dms_matches = list(self._COORD_PATTERN.finditer(text))
        # DMS comes in pairs (lat, lon)
        dms_pairs = []
        i = 0
        while i < len(dms_matches):
            m = dms_matches[i]
            if m.group(1) is not None:  # DMS format
                if i + 1 < len(dms_matches) and dms_matches[i + 1].group(1) is not None:
                    m2 = dms_matches[i + 1]
                    lat = self._dms_to_decimal(float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4))
                    lon = self._dms_to_decimal(float(m2.group(1)), float(m2.group(2)), float(m2.group(3)), m2.group(4))
                    dms_pairs.append((lat, lon))
                    i += 2
                    continue
            i += 1
        coords.extend(dms_pairs)

        # Decimal DMS: 37.75°N, 122.45°W
        dec_dms_matches = re.finditer(
            r"(-?\d{1,3}(?:\.\d+)?)\s*[°dD]\s*([NSEWnsew])", text
        )
        dec_dms_pairs = list(dec_dms_matches)
        i = 0
        while i + 1 < len(dec_dms_pairs):
            m1 = dec_dms_pairs[i]
            m2 = dec_dms_pairs[i + 1]
            val1 = float(m1.group(1))
            val2 = float(m2.group(1))
            dir1 = m1.group(2).upper()
            dir2 = m2.group(2).upper()
            if dir1 in ("N", "S"):
                lat = val1 if dir1 == "N" else -val1
                lon = val2 if dir2 == "E" else -val2
            else:
                lon = val1 if dir1 == "E" else -val1
                lat = val2 if dir2 == "N" else -val2
            coords.append((lat, lon))
            i += 2

        # Plain decimal: lat, lon
        for m in self._DECIMAL_COORD_PATTERN.finditer(text):
            lat = float(m.group(1))
            lon = float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                coords.append((lat, lon))

        return coords

    def extract_durations(self, text: str) -> list[str]:
        """Extract duration expressions as human-readable strings."""
        return [m.group(0) for m in self._DURATION_PATTERN.finditer(text)]

    def normalize(self, text: str) -> str:
        """Normalize an NL command string."""
        # lowercase
        result = text.lower().strip()
        # collapse whitespace
        result = re.sub(r"\s+", " ", result)
        # remove trailing punctuation
        result = result.rstrip(".,;!?")
        # normalize common marine abbreviations
        result = re.sub(r"\bnm\b", "nautical miles", result)
        result = re.sub(r"\bkn\b", "knots", result)
        result = re.sub(r"\bkts?\b", "knots", result)
        result = re.sub(r"\bdeg\b", "degrees", result)
        result = re.sub(r"\bhrs?\b", "hours", result)
        result = re.sub(r"\bmins?\b", "minutes", result)
        result = re.sub(r"\bsecs?\b", "seconds", result)
        result = re.sub(r"\bmph\b", "miles per hour", result)
        result = re.sub(r"\bkph\b", "kilometers per hour", result)
        result = re.sub(r"\blat\b", "latitude", result)
        result = re.sub(r"\blon\b", "longitude", result)
        return result.strip()

    def extract_commands(self, text: str) -> list[str]:
        """Split compound text into individual command strings."""
        # Split on sentence-ending punctuation followed by space
        parts = re.split(r"[.;]\s+", text)
        # Also split on "and then" / "then"
        commands: list[str] = []
        for part in parts:
            sub = re.split(r"\band then\b|\bthen\b", part, flags=re.IGNORECASE)
            commands.extend(s.strip() for s in sub if s.strip())
        return commands

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _find_root_verb(tokens: list[Token]) -> str:
        """Return the lemma of the first VERB token, or empty string."""
        for t in tokens:
            if t.pos_tag == "VERB":
                return t.lemma
        return ""

    @staticmethod
    def _compute_parse_confidence(tokens: list[Token], entities: list[Entity]) -> float:
        """Heuristic confidence score for a parse result."""
        if not tokens:
            return 0.0
        has_verb = any(t.pos_tag == "VERB" for t in tokens)
        base = 0.3
        if has_verb:
            base += 0.4
        if entities:
            base += min(0.3, 0.1 * len(entities))
        return min(1.0, base)

    @staticmethod
    def _dms_to_decimal(deg: float, min_: float, sec: float, direction: str) -> float:
        """Convert DMS to decimal degrees."""
        sign = -1.0 if direction.upper() in ("S", "W") else 1.0
        return sign * (deg + min_ / 60.0 + sec / 3600.0)
