"""NEXUS Agent Runtime - A2A opcode definitions.

29 A2A opcodes for agent-to-agent communication.
All A2A opcodes are NOP on ESP32 firmware (backward compatible).
"""

from __future__ import annotations


def get_a2a_opcodes() -> dict[int, str]:
    """Return a mapping of A2A opcode values to names."""
    return {
        0x20: "DECLARE_INTENT",
        0x21: "ASSERT_GOAL",
        0x22: "VERIFY_OUTCOME",
        0x23: "EXPLAIN_FAILURE",
        0x30: "TELL",
        0x31: "ASK",
        0x32: "DELEGATE",
        0x33: "REPORT_STATUS",
        0x34: "REQUEST_OVERRIDE",
        0x40: "REQUIRE_CAPABILITY",
        0x41: "DECLARE_SENSOR_NEED",
        0x42: "DECLARE_ACTUATOR_USE",
        0x50: "TRUST_CHECK",
        0x51: "AUTONOMY_LEVEL_ASSERT",
        0x52: "SAFE_BOUNDARY",
        0x53: "RATE_LIMIT",
    }
