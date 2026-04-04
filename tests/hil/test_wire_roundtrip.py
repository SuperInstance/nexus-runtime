"""NEXUS HIL Tests - Wire protocol round-trip.

Framework for measuring wire protocol latency between ESP32 and Jetson.
Requires physical RS-422 connection.
"""

from __future__ import annotations


class TestWireRoundtrip:
    """Wire protocol HIL tests (skeleton)."""

    def test_heartbeat_rtt(self) -> None:
        """Measure heartbeat round-trip time (target: <5ms).

        This test requires:
        - ESP32-S3 connected via RS-422 to Jetson
        - Both devices running NEXUS firmware/software
        """
        # TODO: Implement HIL heartbeat RTT measurement
        pass

    def test_reflex_deploy_roundtrip(self) -> None:
        """Measure REFLEX_DEPLOY round-trip latency (target: <30ms)."""
        # TODO: Implement HIL reflex deploy test
        pass

    def test_max_throughput(self) -> None:
        """Measure maximum frame throughput on RS-422 link."""
        # TODO: Implement throughput measurement at 921600 baud
        pass
