"""One-shot keyless-ring rescue for failed keyed/configured web calls.

Stateless by design: a rescue routes THIS call through the free-tier ring (plugins/web/keyless_mcp.py);
the next web_search/web_extract call attempts the chosen backend again. Callers must never cache a
rescue-served response, or the one-shot rescue becomes sticky for a whole TTL. Logs under the origin
(tools.web_tools) logger.
"""

import logging

logger = logging.getLogger("tools.web_tools")

# Ring vendor -> env var holding its paid key (keyed mode ⇒ eligible for rescue).
_RING_KEY_VARS = {
    "exa": "EXA_API_KEY", "parallel": "PARALLEL_API_KEY",
    "firecrawl": "FIRECRAWL_API_KEY", "keenable": "KEENABLE_API_KEY",
}


def _keyless_rescue_enabled() -> bool:
    """``web.keyless_rescue`` (default on), implicitly off when the keyless tier is disabled."""
    from tools.web_tools import _load_web_config
    if not _load_web_config().get("keyless_rescue", True):
        return False
    try:
        from agent.web_search_registry import _keyless_tier_enabled
        return _keyless_tier_enabled()
    except Exception as exc:
        logger.debug("keyless rescue tier check failed: %s", exc)
        return False


def _ring_vendor_keyless(name: str) -> bool:
    """Did *name*'s own provider route this call through the anonymous keyless ring?

    Mirrors the predicate each ring provider evaluates before calling, so eligibility reflects what
    actually happened. Firecrawl owns extra routes that bypass the ring without a key — the managed
    Nous Tool Gateway (persisted ``nous`` selection, or the legacy never-configured fallback when the
    gateway is ready) and a self-hosted ``FIRECRAWL_API_URL`` — so it is asked directly.
    """
    if name == "firecrawl":
        from plugins.web.firecrawl.provider import _use_keyless_ring
        return _use_keyless_ring()
    from agent.web_search_provider import get_provider_env
    from plugins.web.keyless_mcp import use_keyless
    key_var = _RING_KEY_VARS.get(name, "")
    return use_keyless(name, get_provider_env(key_var) if key_var else "")


def _managed_search_fallback(provider, original_error: str, query: str, limit: int):
    """Try managed Firecrawl for this call only; None leaves the original error for keyless rescue.
    Managed Firecrawl is billed, so a caller on free fast search alone never reaches it."""
    from agent.web_search_provider import get_provider_env
    from tools import web_tools as _wt
    if (getattr(provider, "name", "") != "perplexity"
            or get_provider_env("PERPLEXITY_API_KEY") or not _wt._managed_web_search() or not _wt._is_tool_gateway_ready()):
        return None
    logger.warning("web_search managed Perplexity failed (%s); serving this call from managed Firecrawl", (original_error or "")[:200])
    try:
        from agent.web_search_registry import get_provider
        resp = get_provider("firecrawl").search(query, limit)
    except Exception as exc:
        resp = {"success": False, "error": str(exc)}
    if not resp.get("success"):
        logger.warning("managed Firecrawl fallback failed too: %s", str(resp.get("error", ""))[:200])
        return None
    resp.setdefault("data", {}).update(
        fallback_from="managed_primary",
        backend_error=f"Primary managed search failed this call ({(original_error or 'unknown error')[:300]}); "
                      "result served by the managed fallback. The next call will use the primary again.",
    )
    return resp


def _rescue_eligible(provider) -> bool:
    """True when a failed call on *provider* should get a one-shot rescue.

    Eligible: any call that did NOT go through the keyless ring — a non-ring backend, a ring vendor
    in keyed mode, or a ring vendor routed through the managed gateway / a self-hosted instance. A
    ring vendor that walked the ring is NOT eligible: its failure means the ring already failed.
    """
    if not _keyless_rescue_enabled() or provider is None:
        return False
    try:
        from plugins.web.keyless_mcp import _KEYLESS_RING
        name = getattr(provider, "name", "")
        return name not in _KEYLESS_RING or not _ring_vendor_keyless(name)
    except Exception as exc:
        logger.debug("rescue eligibility check failed: %s", exc)
        return False


def _rescue_search(provider_name: str, original_error: str, query: str, limit: int) -> dict:
    """Rescue a failed search via the ring; annotate the result with the original failure."""
    from plugins.web.keyless_mcp import search_with_failover
    logger.warning(
        "web_search backend '%s' failed (%s); one-shot keyless rescue",
        provider_name, (original_error or "")[:200],
    )
    rescued = search_with_failover(provider_name, query, limit)
    if rescued.get("success"):
        rescued.setdefault("data", {}).update(
            rescued_from=provider_name,
            backend_error=(
                f"Configured backend '{provider_name}' failed this call "
                f"({(original_error or 'unknown error')[:300]}); result served by the keyless free tier. "
                f"The next call will use '{provider_name}' again."
            ),
        )
        return rescued
    # Ring also failed: the ORIGINAL error names the user's setup, so lead with it.
    return {
        "success": False,
        "error": (
            f"{original_error or 'search failed'} "
            f"(keyless rescue also failed: {rescued.get('error', 'unknown')})"
        ),
    }


def _configured_fallback_names(kind: str) -> list[str]:
    """Ordered backend names from ``web.<kind>_fallbacks`` (else the shared ``web.backend``).

    Accepts a YAML list, a comma-separated string, or a bracketed string. These are extra
    ladder rungs tried for ONE call when the configured primary fails — the same one-shot,
    never-cached semantics as the keyless rescue, which stays as the last resort behind them.
    Key name matches upstream's ``web.search_fallbacks`` / ``web.extract_fallbacks`` proposal.
    """
    from tools.web_tools import _load_web_config

    cfg = _load_web_config()
    raw = cfg.get(f"{kind}_fallbacks", cfg.get("backend"))
    if isinstance(raw, str):
        text = raw.strip().strip("[]")
        raw = [part for part in text.split(",")] if text else []
    if not isinstance(raw, (list, tuple)):
        return []
    names: list[str] = []
    for item in raw:
        name = str(item or "").strip().strip("\"'").strip().lower()
        if name and name not in names:
            names.append(name)
    return names


def _fallback_search(provider, original_error: str, query: str, limit: int):
    """Serve this call from the first working ``web.search_fallbacks`` rung; None if none worked."""
    from tools import web_tools as _wt

    primary = getattr(provider, "name", "")
    for name in _configured_fallback_names("search"):
        if name == primary:
            continue
        fallback = _wt._registered_web_provider(name)
        if fallback is None or not getattr(fallback, "supports_search", lambda: False)():
            logger.debug("web.search_fallbacks '%s' is not a registered search backend; skipped", name)
            continue
        try:
            resp = fallback.search(query, limit)
        except Exception as exc:
            logger.warning("web.search_fallbacks '%s' raised: %s", name, exc)
            continue
        if not resp.get("success"):
            logger.warning("web.search_fallbacks '%s' failed too: %s", name, str(resp.get("error", ""))[:200])
            continue
        logger.warning(
            "web_search backend '%s' failed (%s); served this call from web.search_fallbacks '%s'",
            primary, (original_error or "")[:200], name,
        )
        resp.setdefault("data", {}).update(
            fallback_backend=name,
            fallback_from=primary,
            backend_error=(
                f"Primary backend '{primary}' failed this call ({(original_error or 'unknown error')[:300]}); "
                f"result served by web.search_fallbacks '{name}'. The next call will use '{primary}' again."
            ),
        )
        return resp
    return None


async def _fallback_extract(provider, urls: list, failed_results: list, *, format=None):
    """Serve a failed extract batch from ``web.extract_fallbacks`` rungs, in order.

    Returns the replacement document list (request order preserved by the provider) or None when
    every rung failed; the caller then hands off to the keyless rescue. Fallback output is never
    cached, matching rescue semantics.
    """
    import asyncio
    import inspect

    from tools import web_tools as _wt

    primary = getattr(provider, "name", "")
    original_error = next((r.get("error") for r in failed_results if r.get("error")), "extract failed")
    for name in _configured_fallback_names("extract"):
        if name == primary:
            continue
        fallback = _wt._registered_web_provider(name)
        if fallback is None or not getattr(fallback, "supports_extract", lambda: False)():
            logger.debug("web.extract_fallbacks '%s' is not a registered extract backend; skipped", name)
            continue
        try:
            if inspect.iscoroutinefunction(fallback.extract):
                docs = await fallback.extract(list(urls), format=format)
            else:
                docs = await asyncio.to_thread(fallback.extract, list(urls), format=format)
        except Exception as exc:
            logger.warning("web.extract_fallbacks '%s' raised: %s", name, exc)
            continue
        docs = [d for d in (docs or []) if isinstance(d, dict)]
        if not any(not d.get("error") for d in docs):
            logger.warning("web.extract_fallbacks '%s' failed for every URL too", name)
            continue
        logger.warning(
            "web_extract backend '%s' failed (%s); served this call from web.extract_fallbacks '%s'",
            primary, (original_error or "")[:200], name,
        )
        for doc in docs:
            meta = doc.setdefault("metadata", {})
            if isinstance(meta, dict):
                meta["fallback_backend"] = name
                meta["fallback_from"] = primary
                meta["backend_error"] = (original_error or "")[:300]
        return docs
    return None


def _policy_blocked_result(result: dict) -> bool:
    """True for a website-policy refusal — intentional, never rescued (it would fetch blocked content)."""
    error = str(result.get("error") or "").lower()
    return bool(result.get("blocked_by_policy")) or "blocked by website policy" in error


def _rescue_extract(provider_name: str, urls: list, results: list) -> list:
    """Rescue a whole-batch extract failure via the ring.

    Only genuine failures are re-fetched; policy-blocked entries are preserved verbatim. If the provider
    broke url/result order parity, every entry is treated as rescueable and the ring's list replaces the
    batch wholesale.
    """
    from plugins.web.keyless_mcp import extract_with_failover

    parity = len(results) == len(urls)
    rescue_idx = [i for i, r in enumerate(results) if not parity or not _policy_blocked_result(r)]
    if not rescue_idx:
        return results  # every failure is an intentional policy block

    rescue_urls = [urls[i] for i in rescue_idx] if parity else list(urls)
    errors = (results[i].get("error") for i in rescue_idx if results[i].get("error"))
    original_error = next(errors, "extract failed")
    logger.warning(
        "web_extract backend '%s' failed all %d URL(s) (%s); one-shot keyless rescue",
        provider_name, len(rescue_urls), (original_error or "")[:200],
    )
    rescued = extract_with_failover(provider_name, list(rescue_urls))
    if rescued and all(r.get("error", "") for r in rescued):
        return results  # rescue also failed everywhere: keep original errors
    for r in rescued:
        meta = None if r.get("error") else r.setdefault("metadata", {})
        if isinstance(meta, dict):
            meta["rescued_from"] = provider_name
            meta["backend_error"] = (original_error or "")[:300]
    if parity and len(rescued) == len(rescue_idx):
        replacements = dict(zip(rescue_idx, rescued))
        return [replacements.get(i, r) for i, r in enumerate(results)]
    return rescued
