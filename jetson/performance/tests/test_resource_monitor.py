"""Tests for jetson.performance.resource_monitor — ResourceSnapshot, ResourceAlert, ResourceMonitor."""

import time
import pytest
from jetson.performance.resource_monitor import (
    ResourceSnapshot, ResourceAlert, ResourceMonitor, Severity,
)

class TestResourceSnapshot:
    def test_construction(self):
        s = ResourceSnapshot(timestamp=100.0, cpu_percent=50.0, memory_used=40.0, memory_total=100.0)
        assert s.cpu_percent == 50.0
        assert s.memory_used == 40.0
        assert s.memory_total == 100.0

    def test_memory_percent(self):
        s = ResourceSnapshot(timestamp=0, cpu_percent=0, memory_used=75.0, memory_total=100.0)
        assert s.memory_percent == pytest.approx(75.0)

    def test_memory_percent_zero_total(self):
        s = ResourceSnapshot(timestamp=0, cpu_percent=0, memory_used=50.0, memory_total=0.0)
        assert s.memory_percent == 0.0

    def test_age(self):
        s = ResourceSnapshot(timestamp=time.time() - 2.0, cpu_percent=0, memory_used=0, memory_total=1)
        assert s.age >= 1.9

    def test_default_values(self):
        s = ResourceSnapshot(timestamp=0, cpu_percent=0, memory_used=0, memory_total=1)
        assert s.disk_io == 0.0
        assert s.network_io == 0.0
        assert s.open_files == 0
        assert s.thread_count == 0

    def test_as_dict(self):
        s = ResourceSnapshot(timestamp=1.0, cpu_percent=25.0, memory_used=50.0, memory_total=100.0, disk_io=10.0, network_io=5.0, open_files=3, thread_count=4)
        d = s.as_dict()
        assert d["cpu_percent"] == 25.0
        assert d["memory_percent"] == 50.0
        assert d["disk_io"] == 10.0
        assert d["open_files"] == 3

class TestResourceAlert:
    def test_construction(self):
        a = ResourceAlert(resource_type="cpu", current_value=95.0, threshold=90.0, severity="high", recommendation="Reduce load")
        assert a.resource_type == "cpu"
        assert a.current_value == 95.0
        assert a.severity == "high"

    def test_as_dict(self):
        a = ResourceAlert(resource_type="mem", current_value=88.0, threshold=85.0, severity="medium", recommendation="GC")
        d = a.as_dict()
        assert set(d.keys()) == {"resource_type","current_value","threshold","severity","recommendation"}

class TestResourceMonitor:
    def test_take_snapshot(self):
        m = ResourceMonitor()
        s = m.take_snapshot()
        assert isinstance(s, ResourceSnapshot)
        assert s.thread_count >= 1

    def test_take_custom_snapshot(self):
        m = ResourceMonitor()
        s = m.take_custom_snapshot(cpu_percent=75.0, memory_used=80.0, memory_total=100.0)
        assert s.cpu_percent == 75.0
        assert s.memory_used == 80.0

    def test_add_snapshot(self):
        m = ResourceMonitor()
        s = ResourceSnapshot(timestamp=1.0, cpu_percent=50.0, memory_used=50.0, memory_total=100.0)
        m.add_snapshot(s)
        assert len(m.get_snapshots()) == 1

    def test_clear_snapshots(self):
        m = ResourceMonitor()
        m.take_snapshot()
        m.take_snapshot()
        m.clear_snapshots()
        assert m.get_snapshots() == []

    def test_register_callback(self):
        m = ResourceMonitor()
        collected = []
        m.register_callback(lambda s: collected.append(s))
        m.take_snapshot()
        assert len(collected) == 1

    def test_compute_trend_empty(self):
        result = ResourceMonitor.compute_trend([], "cpu_percent")
        assert result["direction"] == "flat"

    def test_compute_trend_single(self):
        s = ResourceSnapshot(timestamp=1.0, cpu_percent=50.0, memory_used=50.0, memory_total=100.0)
        result = ResourceMonitor.compute_trend([s], "cpu_percent")
        assert result["direction"] == "flat"

    def test_compute_trend_increasing(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=float(i * 10), memory_used=50.0, memory_total=100.0)
            for i in range(5)
        ]
        result = ResourceMonitor.compute_trend(snapshots, "cpu_percent")
        assert result["direction"] == "increasing"
        assert result["slope"] > 0

    def test_compute_trend_decreasing(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=float(50 - i * 10), memory_used=50.0, memory_total=100.0)
            for i in range(5)
        ]
        result = ResourceMonitor.compute_trend(snapshots, "cpu_percent")
        assert result["direction"] == "decreasing"

    def test_compute_trend_flat(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=50.0, memory_used=50.0, memory_total=100.0)
            for i in range(5)
        ]
        result = ResourceMonitor.compute_trend(snapshots, "cpu_percent")
        assert result["direction"] == "flat"

    def test_compute_trend_memory_percent(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=0, memory_used=float(i * 10), memory_total=100.0)
            for i in range(5)
        ]
        result = ResourceMonitor.compute_trend(snapshots, "memory_percent")
        assert result["direction"] == "increasing"

    def test_compute_trend_disk_io(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=0, memory_used=0, memory_total=1, disk_io=float(i))
            for i in range(5)
        ]
        result = ResourceMonitor.compute_trend(snapshots, "disk_io")
        assert result["direction"] == "increasing"

    def test_detect_anomalies_none(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=50.0, memory_used=50.0, memory_total=100.0)
            for i in range(10)
        ]
        anomalies = ResourceMonitor.detect_anomalies(snapshots)
        assert anomalies == []

    def test_detect_anomalies_present(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=50.0, memory_used=50.0, memory_total=100.0)
            for i in range(10)
        ]
        # Insert anomaly
        snapshots[5] = ResourceSnapshot(timestamp=5.0, cpu_percent=99.0, memory_used=50.0, memory_total=100.0)
        anomalies = ResourceMonitor.detect_anomalies(snapshots)
        assert len(anomalies) >= 1

    def test_detect_anomalies_insufficient(self):
        snapshots = [ResourceSnapshot(timestamp=1.0, cpu_percent=50.0, memory_used=50.0, memory_total=100.0)]
        assert ResourceMonitor.detect_anomalies(snapshots) == []

    def test_predict_exhaustion_memory(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=0, memory_used=float(10 + i * 10), memory_total=100.0)
            for i in range(5)
        ]
        result = ResourceMonitor.predict_exhaustion(snapshots, "memory")
        assert result["time_until_exhaustion"] is not None
        assert result["slope"] > 0

    def test_predict_exhaustion_stable(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=0, memory_used=50.0, memory_total=100.0)
            for i in range(5)
        ]
        result = ResourceMonitor.predict_exhaustion(snapshots, "memory")
        assert result["time_until_exhaustion"] is None

    def test_predict_exhaustion_insufficient(self):
        result = ResourceMonitor.predict_exhaustion([], "memory")
        assert result["time_until_exhaustion"] is None

    def test_predict_exhaustion_already_exhausted(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=0, memory_used=100.0, memory_total=100.0)
            for i in range(3)
        ]
        result = ResourceMonitor.predict_exhaustion(snapshots, "memory")
        assert result["time_until_exhaustion"] == 0.0

    def test_predict_exhaustion_cpu(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=float(10 + i * 10), memory_used=0, memory_total=1)
            for i in range(5)
        ]
        result = ResourceMonitor.predict_exhaustion(snapshots, "cpu")
        assert result["resource_type"] == "cpu"

    def test_generate_alerts_empty(self):
        assert ResourceMonitor.generate_alerts([]) == []

    def test_generate_alerts_cpu(self):
        snapshots = [ResourceSnapshot(timestamp=1.0, cpu_percent=95.0, memory_used=50.0, memory_total=100.0)]
        alerts = ResourceMonitor.generate_alerts(snapshots)
        types = [a.resource_type for a in alerts]
        assert "cpu_percent" in types

    def test_generate_alerts_memory(self):
        snapshots = [ResourceSnapshot(timestamp=1.0, cpu_percent=50.0, memory_used=90.0, memory_total=100.0)]
        alerts = ResourceMonitor.generate_alerts(snapshots)
        types = [a.resource_type for a in alerts]
        assert "memory_percent" in types

    def test_generate_alerts_disk(self):
        snapshots = [ResourceSnapshot(timestamp=1.0, cpu_percent=50.0, memory_used=50.0, memory_total=100.0, disk_io=90.0)]
        alerts = ResourceMonitor.generate_alerts(snapshots)
        types = [a.resource_type for a in alerts]
        assert "disk_io" in types

    def test_generate_alerts_network(self):
        snapshots = [ResourceSnapshot(timestamp=1.0, cpu_percent=50.0, memory_used=50.0, memory_total=100.0, network_io=85.0)]
        alerts = ResourceMonitor.generate_alerts(snapshots)
        types = [a.resource_type for a in alerts]
        assert "network_io" in types

    def test_generate_alerts_no_breach(self):
        snapshots = [ResourceSnapshot(timestamp=1.0, cpu_percent=10.0, memory_used=20.0, memory_total=100.0)]
        alerts = ResourceMonitor.generate_alerts(snapshots)
        assert alerts == []

    def test_generate_alerts_custom_thresholds(self):
        snapshots = [ResourceSnapshot(timestamp=1.0, cpu_percent=50.0, memory_used=50.0, memory_total=100.0)]
        alerts = ResourceMonitor.generate_alerts(snapshots, {"cpu_percent": 40.0})
        types = [a.resource_type for a in alerts]
        assert "cpu_percent" in types

    def test_generate_alerts_critical_cpu(self):
        snapshots = [ResourceSnapshot(timestamp=1.0, cpu_percent=99.5, memory_used=50.0, memory_total=100.0)]
        alerts = ResourceMonitor.generate_alerts(snapshots)
        cpu_alert = next(a for a in alerts if a.resource_type == "cpu_percent")
        assert cpu_alert.severity == "critical"

    def test_compute_resource_efficiency_empty(self):
        assert ResourceMonitor.compute_resource_efficiency([]) == {}

    def test_compute_resource_efficiency_balanced(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=60.0, memory_used=60.0, memory_total=100.0)
            for i in range(5)
        ]
        eff = ResourceMonitor.compute_resource_efficiency(snapshots)
        assert "cpu" in eff
        assert "memory" in eff
        assert eff["cpu"] == pytest.approx(100.0, abs=1)
        assert eff["memory"] == pytest.approx(100.0, abs=1)

    def test_compute_resource_efficiency_overloaded(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=100.0, memory_used=100.0, memory_total=100.0)
            for i in range(5)
        ]
        eff = ResourceMonitor.compute_resource_efficiency(snapshots)
        assert eff["cpu"] < 100.0

    def test_compute_resource_efficiency_underutilized(self):
        snapshots = [
            ResourceSnapshot(timestamp=float(i), cpu_percent=5.0, memory_used=5.0, memory_total=100.0)
            for i in range(5)
        ]
        eff = ResourceMonitor.compute_resource_efficiency(snapshots)
        assert eff["cpu"] < 100.0
        assert eff["cpu"] > 0.0

    def test_monitor_interval(self):
        m = ResourceMonitor()
        collected = []
        snaps = m.monitor_interval(0.1, lambda s: collected.append(s))
        assert len(snaps) >= 1
        assert len(collected) >= 1

class TestSeverity:
    def test_values(self):
        assert Severity.LOW.value == "low"
        assert Severity.CRITICAL.value == "critical"
