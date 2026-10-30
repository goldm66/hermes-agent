"""Configured web fallback ladder: ``web.search_fallbacks`` / ``web.extract_fallbacks``.

A ladder rung sits between the configured primary and the keyless rescue: it is tried for ONE
call, never cached, and the result carries provenance so the model knows a backup served it.

Covers:
- key parsing: YAML list, comma string, bracketed string, shared ``web.backend`` fallback,
  dedupe/lowercase, junk types
- search: primary failure walks the rungs in order; skipped rungs that are unregistered or
  lack the capability; annotation (fallback_backend / fallback_from / backend_error); the
  ladder failing hands off to the keyless rescue; a healthy primary never touches the ladder
- extract: whole-batch failure walks the rungs; partial failure does not; annotation survives
  ``_trim_results`` (the trim rebuild used to drop every metadata key)
"""

import json
from unittest.mock import patch

import pytest

from tools import web_tools, web_tools_rescue
from tools.web_tools_truncate import _trim_results
from plugins.web import keyless_mcp


class _Provider:
    """Minimal provider double: configurable name and failure behaviour."""

    def __init__(self, name, *, fail=True, text="body"):
        self.name = name
        self.display_name = name
        self.fail = fail
        self.text = text
        self.search_calls = 0
        self.extract_calls = 0

    def supports_search(self):
        return True

    def supports_extract(self):
        return True

    def is_available(self):
        return True

    def search(self, query, limit=5):
        self.search_calls += 1
        if self.fail:
            return {"success": False, "error": f"{self.name} exploded"}
        return {
            "success": True,
            "data": {"web": [{"url": f"https://{self.name}.example", "title": self.name}]},
        }

    def extract(self, urls, **kwargs):
        self.extract_calls += 1
        if self.fail:
            return [
                {"url": u, "title": "", "content": "", "error": f"{self.name} exploded"}
                for u in urls
            ]
        return [
            {"url": u, "title": self.name, "content": self.text,
             "raw_content": self.text, "metadata": {"sourceURL": u, "fetcher": self.name}}
            for u in urls
        ]


def _registry(*providers):
    by_name = {p.name: p for p in providers}

    def _get(name):
        return by_name.get(name)

    return _get


class TestConfiguredNames:
    @pytest.mark.parametrize(
        "cfg, expected",
        [
            ({"search_fallbacks": ["deepseek"]}, ["deepseek"]),
            ({"search_fallbacks": "deepseek"}, ["deepseek"]),
            ({"search_fallbacks": "deepseek, ddgs"}, ["deepseek", "ddgs"]),
            ({"search_fallbacks": "[deepseek, ddgs]"}, ["deepseek", "ddgs"]),
            ({"search_fallbacks": [" DeepSeek ", "deepseek", "", None]}, ["deepseek"]),
            ({"search_fallbacks": []}, []),
            ({"search_fallbacks": 7}, []),
            ({}, []),
            # Shared web.backend is the documented last-resort name source.
            ({"backend": "ddgs"}, ["ddgs"]),
            # A per-capability list wins over the shared key.
            ({"backend": "ddgs", "search_fallbacks": ["deepseek"]}, ["deepseek"]),
        ],
    )
    def test_parsing(self, monkeypatch, cfg, expected):
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: cfg)
        assert web_tools_rescue._configured_fallback_names("search") == expected

    def test_extract_key_is_separate(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"search_fallbacks": ["a"], "extract_fallbacks": ["b"]},
        )
        assert web_tools_rescue._configured_fallback_names("extract") == ["b"]


class TestSearchLadder:
    def _dispatch(self, monkeypatch, primary, *others):
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(
            "agent.web_search_registry.get_provider", _registry(primary, *others)
        )
        return json.loads(web_tools.web_search_tool("q", limit=2))

    def test_fallback_serves_and_annotates(self, monkeypatch):
        primary = _Provider("boom")
        backup = _Provider("deepseek", fail=False)
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"search_backend": "boom", "search_fallbacks": ["deepseek"]},
        )
        with patch.object(keyless_mcp, "search_with_failover") as ring:
            out = self._dispatch(monkeypatch, primary, backup)
        assert out["success"] is True
        assert out["data"]["fallback_backend"] == "deepseek"
        assert out["data"]["fallback_from"] == "boom"
        assert "boom exploded" in out["data"]["backend_error"]
        assert "next call" in out["data"]["backend_error"].lower()
        ring.assert_not_called()  # the configured rung wins over the keyless ring

    def test_ladder_is_walked_in_order(self, monkeypatch):
        primary = _Provider("boom")
        dead = _Provider("dead")
        alive = _Provider("deepseek", fail=False)
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"search_backend": "boom", "search_fallbacks": ["dead", "deepseek"]},
        )
        with patch.object(keyless_mcp, "search_with_failover") as ring:
            out = self._dispatch(monkeypatch, primary, dead, alive)
        assert out["data"]["fallback_backend"] == "deepseek"
        assert dead.search_calls == 1 and alive.search_calls == 1
        ring.assert_not_called()

    def test_unregistered_rung_is_skipped(self, monkeypatch):
        primary = _Provider("boom")
        alive = _Provider("deepseek", fail=False)
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"search_backend": "boom", "search_fallbacks": ["ghost", "deepseek"]},
        )
        out = self._dispatch(monkeypatch, primary, alive)
        assert out["data"]["fallback_backend"] == "deepseek"

    def test_primary_itself_is_never_a_rung(self, monkeypatch):
        primary = _Provider("boom")
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"search_backend": "boom", "search_fallbacks": ["boom"]},
        )
        with patch.object(
            keyless_mcp, "search_with_failover",
            return_value={"success": True, "data": {"web": []}},
        ):
            out = self._dispatch(monkeypatch, primary)
        assert primary.search_calls == 1  # no self-retry through the ladder

    def test_healthy_primary_never_touches_ladder(self, monkeypatch):
        primary = _Provider("boom", fail=False)
        backup = _Provider("deepseek", fail=False)
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"search_backend": "boom", "search_fallbacks": ["deepseek"]},
        )
        out = self._dispatch(monkeypatch, primary, backup)
        assert out["success"] is True
        assert backup.search_calls == 0
        assert "fallback_backend" not in out["data"]

    def test_dead_ladder_falls_through_to_rescue(self, monkeypatch):
        primary = _Provider("boom")
        dead = _Provider("dead")
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"search_backend": "boom", "search_fallbacks": ["dead"]},
        )
        with patch.object(
            keyless_mcp, "search_with_failover",
            return_value={"success": True, "data": {"web": []}},
        ) as ring:
            out = self._dispatch(monkeypatch, primary, dead)
        ring.assert_called_once()
        assert out["success"] is True


class TestExtractLadder:
    async def _dispatch(self, monkeypatch, primary, *others, urls=("https://a", "https://b")):
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(
            "agent.web_search_registry.get_provider", _registry(primary, *others)
        )

        async def _allow_all(url, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
        raw = await web_tools.web_extract_tool(list(urls))
        data = json.loads(raw)
        return data["results"] if isinstance(data, dict) and "results" in data else data

    @pytest.mark.asyncio
    async def test_fallback_serves_and_annotates_through_trim(self, monkeypatch):
        primary = _Provider("boom")
        backup = _Provider("deepseek", fail=False, text="hello world")
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"extract_backend": "boom", "extract_fallbacks": ["deepseek"]},
        )
        with patch.object(keyless_mcp, "extract_with_failover") as ring:
            results = await self._dispatch(monkeypatch, primary, backup)
        assert all(not r.get("error") for r in results)
        assert results[0]["content"].startswith("hello world")
        # Provenance must survive the trim rebuild — the regression this covers.
        assert results[0]["metadata"]["fallback_backend"] == "deepseek"
        assert results[0]["metadata"]["fallback_from"] == "boom"
        assert "boom exploded" in results[0]["metadata"]["backend_error"]
        ring.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_walk_ladder(self, monkeypatch):
        class _Partial(_Provider):
            def extract(self, urls, **kwargs):
                self.extract_calls += 1
                return [
                    {"url": urls[0], "title": "A", "content": "fine",
                     "raw_content": "fine", "metadata": {}},
                    {"url": urls[1], "title": "", "content": "", "error": "404"},
                ]

        backup = _Provider("deepseek", fail=False)
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"extract_backend": "partial", "extract_fallbacks": ["deepseek"]},
        )
        results = await self._dispatch(monkeypatch, _Partial("partial"), backup)
        assert results[1].get("error")
        assert backup.extract_calls == 0

    @pytest.mark.asyncio
    async def test_dead_ladder_falls_through_to_rescue(self, monkeypatch):
        primary = _Provider("boom")
        dead = _Provider("dead")
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"extract_backend": "boom", "extract_fallbacks": ["dead"]},
        )
        good = [{"url": "https://a", "title": "A", "content": "x", "raw_content": "x",
                 "metadata": {}}]
        with patch.object(keyless_mcp, "extract_with_failover", return_value=good) as ring:
            results = await self._dispatch(monkeypatch, primary, dead, urls=("https://a",))
        ring.assert_called_once()
        assert not results[0].get("error")


class TestTrimKeepsProvenance:
    def test_metadata_provenance_survives(self):
        trimmed = _trim_results([
            {"url": "https://a", "title": "A", "content": "x",
             "metadata": {"fallback_backend": "deepseek", "fallback_from": "boom",
                          "backend_error": "boom exploded", "sourceURL": "https://a"}},
        ])
        assert trimmed[0]["metadata"] == {
            "fallback_backend": "deepseek", "fallback_from": "boom", "backend_error": "boom exploded",
        }

    def test_empty_metadata_is_not_emitted(self):
        trimmed = _trim_results([{"url": "https://a", "title": "A", "content": "x",
                                  "metadata": {"sourceURL": "https://a"}}])
        assert "metadata" not in trimmed[0]

    def test_no_metadata_key_still_trimmed(self):
        trimmed = _trim_results([{"url": "https://a", "title": "A", "content": "x"}])
        assert trimmed[0] == {"url": "https://a", "title": "A", "content": "x", "error": None}
