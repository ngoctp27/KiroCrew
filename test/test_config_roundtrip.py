"""Tests that KiroCrewConfig.save() preserves all dataclass fields.

Regression test for the bug where to_dict() omitted secretary,
taskrunner, orchestrator, skills, and tunnel — causing save() to
silently drop them from config.json.
"""

from __future__ import annotations

import json
from dataclasses import fields
from unittest.mock import patch

import pytest

from kiro_crew.config.loader import KiroCrewConfig


@pytest.fixture()
def cfg_file(tmp_path):
    """Redirect config_path() to a temp file for isolation."""
    p = tmp_path / "config.json"
    p.write_text("{}", encoding="utf-8")
    with patch("kiro_crew.config.loader.config_path", return_value=p):
        yield p


def test_to_dict_includes_all_dataclass_fields():
    """Every public field on KiroCrewConfig must appear in to_dict() output."""
    cfg = KiroCrewConfig()
    d = cfg.to_dict()
    # Fields that are serialized under a different key or merged into slack
    SPECIAL = {
        "slack_channels",
        "slack_dm_activation",
        "slack_channel_default_activation",
        "observe_max_messages",
        "observe_ttl_hours",
    }
    for f in fields(KiroCrewConfig):
        if f.name in SPECIAL:
            continue
        if f.name.startswith("_"):
            # Private load-status fields (_extra_sections, _degraded_sections)
            # are deliberately not round-tripped: they describe THIS load, not
            # user configuration.
            continue
        assert f.name in d, f"to_dict() missing field: {f.name}"


def test_save_load_roundtrip_taskrunner(cfg_file):
    """TaskRunner config must survive a save/load cycle."""
    cfg = KiroCrewConfig()
    cfg.taskrunner.max_parallel_steps = 5
    cfg.save()

    raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert raw["taskrunner"]["max_parallel_steps"] == 5


def test_save_load_roundtrip_orchestrator(cfg_file):
    """Orchestrator config must survive a save/load cycle."""
    cfg = KiroCrewConfig()
    cfg.orchestrator.stage_timeout_seconds = 900
    cfg.save()

    raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert raw["orchestrator"]["stage_timeout_seconds"] == 900


def test_save_load_roundtrip_skills(cfg_file):
    """Skills config must survive a save/load cycle."""
    cfg = KiroCrewConfig()
    cfg.skills.max_triggered = 5
    cfg.save()

    raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert raw["skills"]["max_triggered"] == 5


def test_save_load_roundtrip_tunnel(cfg_file):
    """Tunnel config must survive a save/load cycle."""
    cfg = KiroCrewConfig()
    cfg.tunnel.enabled = True
    cfg.save()

    raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert raw["tunnel"]["enabled"] is True


def _load_slack(cfg_file, slack: dict) -> KiroCrewConfig:
    from kiro_crew.config.loader import _invalidate_config_cache

    cfg_file.write_text(json.dumps({"slack": slack}), encoding="utf-8")
    _invalidate_config_cache()
    return KiroCrewConfig.load()


def test_channel_default_activation_roundtrips_and_default_is_not_emitted(cfg_file):
    """``off`` survives load -> to_dict -> load; the default ``mention`` is never written."""
    cfg = _load_slack(cfg_file, {"channel_default_activation": "off"})
    emitted = cfg.to_dict()["slack"]
    assert emitted["channel_default_activation"] == "off"
    assert _load_slack(cfg_file, emitted).slack_channel_default_activation == "off"

    assert "channel_default_activation" not in KiroCrewConfig().to_dict()["slack"]


def test_save_does_not_resurrect_a_cleared_channel_default_activation(cfg_file):
    """A key loaded as ``off`` must not come back via the unknown-key capture once reset."""
    cfg = _load_slack(cfg_file, {"channel_default_activation": "off"})
    cfg.slack_channel_default_activation = "mention"
    cfg.save()
    raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert "channel_default_activation" not in raw.get("slack", {})


def test_copy_slack_fields_carries_channel_default_activation():
    from kiro_crew.slack.handler import copy_slack_fields

    fresh, live = KiroCrewConfig(), KiroCrewConfig()
    fresh.slack_channel_default_activation = "off"
    copy_slack_fields(fresh, live)
    assert live.slack_channel_default_activation == "off"


def test_reload_orch_cfg_carries_channel_default_activation(monkeypatch):
    import kiro_crew.slack.handler as h

    live, fresh = KiroCrewConfig(), KiroCrewConfig()
    fresh.slack_channel_default_activation = "off"
    monkeypatch.setattr(h, "_orch_cfg", live)
    h._reload_orch_cfg(fresh)
    assert live.slack_channel_default_activation == "off"


def test_save_load_roundtrip_channel_context(cfg_file):
    """Static channel context must survive a save/load cycle (auto via asdict)."""
    from kiro_crew.config.loader import ChannelConfig, _invalidate_config_cache

    cfg = KiroCrewConfig()
    cfg.slack_channels["C0X"] = ChannelConfig(
        channel_name="ops", channel_topic="Alerts", channel_description="Balance"
    )
    cfg.save()

    raw = json.loads(cfg_file.read_text(encoding="utf-8"))["slack"]["channels"]["C0X"]
    assert raw["channel_name"] == "ops"
    assert raw["channel_topic"] == "Alerts"
    assert raw["channel_description"] == "Balance"

    _invalidate_config_cache()
    try:
        ch = KiroCrewConfig.load().channel_config("C0X")
    finally:
        _invalidate_config_cache()
    assert (ch.channel_name, ch.channel_topic, ch.channel_description) == (
        "ops",
        "Alerts",
        "Balance",
    )
