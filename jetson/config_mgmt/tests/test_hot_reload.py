"""Tests for hot_reload.py — ReloadEvent, ConfigWatcher."""

import json
import os
import time
import pytest
import tempfile

from jetson.config_mgmt.hot_reload import ReloadEvent, ConfigWatcher


class TestReloadEvent:
    def test_creation_defaults(self):
        event = ReloadEvent(timestamp=1000.0, config_name="test")
        assert event.timestamp == 1000.0
        assert event.config_name == "test"
        assert event.changes == {}
        assert event.source == "file"

    def test_creation_with_changes(self):
        event = ReloadEvent(
            timestamp=2000.0,
            config_name="app",
            changes={"added": {"x": 1}, "removed": {}, "changed": {"port": {"old": 80, "new": 443}}},
            source="reload",
        )
        assert event.config_name == "app"
        assert event.source == "reload"
        assert "x" in event.changes["added"]

    def test_event_is_dataclass(self):
        event = ReloadEvent(timestamp=0.0, config_name="x")
        # dataclass instances support attribute access
        assert hasattr(event, "timestamp")
        assert hasattr(event, "config_name")
        assert hasattr(event, "changes")
        assert hasattr(event, "source")


class TestConfigWatcher:
    def _write_json(self, data, path=None):
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".json")
            os.write(fd, json.dumps(data).encode())
            os.close(fd)
            return path
        else:
            with open(path, "w") as f:
                json.dump(data, f)
            return path

    def test_watch_creates_config(self):
        path = self._write_json({"host": "localhost"})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            assert name == os.path.splitext(os.path.basename(path))[0]
            config = watcher.get_current_config(name)
            assert config["host"] == "localhost"
        finally:
            os.unlink(path)

    def test_watch_registers_callback(self):
        path = self._write_json({"a": 1})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            assert len(watcher._subscriptions.get(name, [])) == 0  # no callback by default

            watcher.watch(path, callback=lambda e: None)
            assert len(watcher._subscriptions.get(name, [])) == 1
        finally:
            os.unlink(path)

    def test_get_current_config_not_loaded(self):
        watcher = ConfigWatcher()
        with pytest.raises(KeyError):
            watcher.get_current_config("nonexistent")

    def test_get_current_config_returns_copy(self):
        path = self._write_json({"key": "value"})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            config1 = watcher.get_current_config(name)
            config1["key"] = "modified"
            config2 = watcher.get_current_config(name)
            assert config2["key"] == "value"
        finally:
            os.unlink(path)

    def test_check_changes_no_change(self):
        path = self._write_json({"key": "value"})
        try:
            watcher = ConfigWatcher()
            watcher.watch(path)
            event = watcher.check_changes()
            assert event is None
        finally:
            os.unlink(path)

    def test_check_changes_with_change(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, json.dumps({"version": 1}).encode())
        os.close(fd)
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            # Ensure mtime difference
            time.sleep(0.05)
            self._write_json({"version": 2}, path)
            event = watcher.check_changes()
            assert event is not None
            assert event.config_name == name
            assert "version" in event.changes.get("changed", {})
        finally:
            os.unlink(path)

    def test_reload_force(self):
        path = self._write_json({"a": 1})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            self._write_json({"a": 2, "b": 3}, path)
            # Force reload
            new_config = watcher.reload(name)
            assert new_config["a"] == 2
            assert new_config["b"] == 3
        finally:
            os.unlink(path)

    def test_reload_not_watched(self):
        watcher = ConfigWatcher()
        with pytest.raises(KeyError):
            watcher.reload("not-watched")

    def test_reload_triggers_history(self):
        path = self._write_json({"v": 1})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            self._write_json({"v": 2}, path)
            watcher.reload(name)
            history = watcher.get_change_history(name)
            assert len(history) >= 1
            assert history[0].config_name == name
            assert history[0].source == "reload"
        finally:
            os.unlink(path)

    def test_subscribe_and_notify(self):
        path = self._write_json({"x": 1})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            received = []
            watcher.subscribe(name, lambda e: received.append(e))
            self._write_json({"x": 2}, path)
            watcher.reload(name)
            assert len(received) == 1
            assert received[0].config_name == name
        finally:
            os.unlink(path)

    def test_subscribe_returns_id(self):
        watcher = ConfigWatcher()
        sub_id = watcher.subscribe("config", lambda e: None)
        assert isinstance(sub_id, int)

    def test_unsubscribe(self):
        path = self._write_json({"a": 1})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            sub_id = watcher.subscribe(name, lambda e: None)
            watcher.unsubscribe(sub_id)
            assert len(watcher._subscriptions.get(name, [])) == 0
        finally:
            os.unlink(path)

    def test_unsubscribe_invalid_id(self):
        watcher = ConfigWatcher()
        with pytest.raises(KeyError):
            watcher.unsubscribe(99999)

    def test_get_change_history_empty(self):
        watcher = ConfigWatcher()
        assert watcher.get_change_history("nonexistent") == []

    def test_get_change_history_limit(self):
        path = self._write_json({"v": 0})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            for i in range(1, 6):
                time.sleep(0.01)
                self._write_json({"v": i}, path)
                watcher.reload(name)
            history = watcher.get_change_history(name, limit=3)
            assert len(history) == 3
        finally:
            os.unlink(path)

    def test_get_change_history_most_recent_first(self):
        path = self._write_json({"v": 1})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            self._write_json({"v": 2}, path)
            watcher.reload(name)
            time.sleep(0.01)
            self._write_json({"v": 3}, path)
            watcher.reload(name)
            history = watcher.get_change_history(name)
            assert len(history) == 2
            # Most recent first: v=3 event first
            assert history[0].changes["changed"]["v"]["new"] == 3
            assert history[1].changes["changed"]["v"]["new"] == 2
        finally:
            os.unlink(path)

    def test_compute_config_fingerprint(self):
        watcher = ConfigWatcher()
        fp1 = watcher.compute_config_fingerprint({"a": 1, "b": 2})
        fp2 = watcher.compute_config_fingerprint({"b": 2, "a": 1})  # same content, different order
        assert fp1 == fp2
        assert len(fp1) == 16  # truncated SHA-256

    def test_compute_config_fingerprint_differs(self):
        watcher = ConfigWatcher()
        fp1 = watcher.compute_config_fingerprint({"a": 1})
        fp2 = watcher.compute_config_fingerprint({"a": 2})
        assert fp1 != fp2

    def test_reload_event_change_detection_added(self):
        path = self._write_json({"a": 1})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            self._write_json({"a": 1, "b": 2}, path)
            event_data = watcher.reload(name)
            # Check internal history
            history = watcher.get_change_history(name, limit=1)
            assert "b" in history[0].changes["added"]
        finally:
            os.unlink(path)

    def test_reload_event_change_detection_removed(self):
        path = self._write_json({"a": 1, "b": 2})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            self._write_json({"a": 1}, path)
            watcher.reload(name)
            history = watcher.get_change_history(name, limit=1)
            assert "b" in history[0].changes["removed"]
        finally:
            os.unlink(path)

    def test_check_changes_nonexistent_file(self):
        watcher = ConfigWatcher()
        event = watcher.check_changes()
        assert event is None

    def test_multiple_watches(self):
        p1 = self._write_json({"name": "first"})
        p2 = self._write_json({"name": "second"})
        try:
            watcher = ConfigWatcher()
            n1 = watcher.watch(p1)
            n2 = watcher.watch(p2)
            assert watcher.get_current_config(n1)["name"] == "first"
            assert watcher.get_current_config(n2)["name"] == "second"
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_subscriber_error_does_not_propagate(self):
        path = self._write_json({"v": 1})
        try:
            watcher = ConfigWatcher()
            name = watcher.watch(path)
            watcher.subscribe(name, lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
            # Should not raise
            self._write_json({"v": 2}, path)
            new_config = watcher.reload(name)
            assert new_config["v"] == 2
        finally:
            os.unlink(path)
