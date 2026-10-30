"""DeepSeek Search + Fetch web provider — bundled, auto-loaded.

Search (``/v1/tools/search``) and page fetch (``/v1/tools/fetch``) are independent
per-request-billed API endpoints; ``search`` is sync, ``extract`` is a coroutine the
dispatcher awaits.
"""

from __future__ import annotations

from plugins.web.deepseek.provider import DeepSeekWebProvider


def register(ctx) -> None:
    """Register the DeepSeek provider with the plugin context."""
    ctx.register_web_search_provider(DeepSeekWebProvider())
