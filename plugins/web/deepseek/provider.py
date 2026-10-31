"""DeepSeek Search + Fetch — plugin form.

Two capabilities from DeepSeek's dedicated tools endpoints (raw JSON over the same
Bearer auth as the model API):

- ``supports_search()``  -> True  (``POST /v1/tools/search``, DeepSeek's own index)
- ``supports_extract()`` -> True  (``POST /v1/tools/fetch``, page -> Markdown)

Both are billed per successful request (¥5 / 1,000 at launch), not per token, so a
rung on a fallback ladder costs a fraction of a cent rather than a model call.

Config keys this provider responds to::

    web:
      search_backend: "deepseek"   # explicit per-capability
      extract_backend: "deepseek"  # explicit per-capability
      backend: "deepseek"          # shared fallback for both

Env vars::

    DEEPSEEK_API_KEY=...     # required
    DEEPSEEK_BASE_URL=...    # optional override of https://api.deepseek.com

No anonymous tier: without a key the provider reports unavailable and the registry
falls through to the next capable backend.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from plugins.web._common import (
    BaseWebSearchProvider, document, page_error, run_search, setup_schema, title_hit,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.deepseek.com"
_KEY_URL = "https://platform.deepseek.com/api_keys"
_MAX_RESULTS = 50  # Search API: count is 1..50
_SEARCH_TIMEOUT = 30
_FETCH_TIMEOUT = 90
_FETCH_CONCURRENCY = 8

# DeepSeek's Fetch returns Markdown with every ASCII punctuation character
# backslash-escaped ("\# Heading", "a \- b", "\|" table cells), which breaks the
# heading/table/list structure web_extract's consumers expect.
_MD_ESCAPE_RE = re.compile(r"\\([!-/:-@\[-`{-~])")


def _missing_key_error() -> str:
    return f"DEEPSEEK_API_KEY is not set. Get a key at {_KEY_URL}"


def _base_url() -> str:
    from agent.web_search_provider import get_provider_env

    return (get_provider_env("DEEPSEEK_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")


def _post_json(path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """POST to a DeepSeek tools endpoint; raise ``ValueError`` carrying the response body.

    The vendor's own error text ("'query' must not be empty", auth/quota failures) is
    what the model needs to see, so non-2xx bodies are re-raised verbatim.
    """
    from agent.web_search_provider import get_provider_env

    api_key = get_provider_env("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(_missing_key_error())
    response = httpx.post(
        f"{_base_url()}{path}",
        json=payload,
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    if response.status_code >= 400:
        raise ValueError((response.text or "").strip() or f"HTTP {response.status_code}")
    return response.json()


def _unescape_markdown(text: str) -> str:
    """Undo DeepSeek's punctuation escaping outside fenced code blocks.

    Fences are left byte-for-byte: real code carries meaningful backslashes
    (``\\n``, ``\\"``) that a blanket rewrite would corrupt.
    """
    if "\\" not in text:
        return text
    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        out.append(line if in_fence else _MD_ESCAPE_RE.sub(r"\1", line))
    return "\n".join(out)


def _description(result: dict[str, Any]) -> str:
    """Short blurb for a search row: the vendor snippet, else a trimmed summary."""
    snippet = str(result.get("snippet") or "").strip()
    if snippet:
        return snippet
    summary = str(result.get("summary") or "").strip()
    return summary[:300]


def _normalize_result(result: dict[str, Any], position: int) -> dict[str, Any]:
    row = title_hit(
        str(result.get("title") or ""),
        str(result.get("url") or ""),
        _description(result),
        position,
    )
    if result.get("date"):
        row["date"] = str(result["date"])
    return row


class DeepSeekWebProvider(BaseWebSearchProvider):
    """Search + extract over ``api.deepseek.com/v1/tools/{search,fetch}``."""

    NAME = "deepseek"
    DISPLAY_NAME = "DeepSeek Search + Fetch"
    KEY_ENV = "DEEPSEEK_API_KEY"
    EXTRACT = True

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            self.DISPLAY_NAME,
            "paid",
            "Per request (¥5 / 1k)",
            self.KEY_ENV,
            "DeepSeek API key",
            _KEY_URL,
        )

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        count = min(max(int(limit or 5), 1), _MAX_RESULTS)

        def _body() -> dict[str, Any]:
            payload = {"query": query, "count": count}
            if self._need_content():
                payload["need_content"] = True
            data = _post_json("/v1/tools/search", payload, _SEARCH_TIMEOUT)
            rows = [_normalize_result(r, i + 1) for i, r in enumerate(data.get("results") or [])]
            return {"success": True, "data": {"web": rows}}

        return run_search("DeepSeek", logger, _body)

    def _need_content(self) -> bool:
        """``web.deepseek_search_content`` — inline page text in search results (off by default)."""
        try:
            from tools.web_tools import _load_web_config

            return bool(_load_web_config().get("deepseek_search_content", False))
        except Exception:
            return False

    async def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        from agent.web_search_provider import get_provider_env

        api_key = get_provider_env("DEEPSEEK_API_KEY")
        urls = [str(u) for u in urls]
        if not api_key:
            return [page_error(u, _missing_key_error()) for u in urls]

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        semaphore = asyncio.Semaphore(_FETCH_CONCURRENCY)

        async with httpx.AsyncClient(
            base_url=_base_url(), timeout=_FETCH_TIMEOUT, headers=headers
        ) as client:

            async def _one(url: str) -> dict[str, Any]:
                async with semaphore:
                    try:
                        response = await client.post("/v1/tools/fetch", json={"url": url})
                        if response.status_code >= 400:
                            raise ValueError((response.text or "").strip() or f"HTTP {response.status_code}")
                        data = response.json()
                    except Exception as exc:
                        logger.warning("DeepSeek fetch failed for %s: %s", url, exc)
                        return page_error(url, f"DeepSeek fetch failed: {exc}")
                content = _unescape_markdown(str(data.get("content") or ""))
                title = str(data.get("title") or "")
                if not content:
                    return page_error(url, "DeepSeek fetch returned an empty page")
                entry = document(url, title, content)
                entry["metadata"]["fetcher"] = "deepseek"
                return entry

            return list(await asyncio.gather(*(_one(u) for u in urls)))
