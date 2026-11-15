# Resolve local actor settings without constructing a provider client.

from __future__ import annotations
import os
from urllib.parse import urlsplit
from ...llm import load_env

DEFAULT_MODEL = "google/gemini-3.5-flash-lite"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


# Treat empty template values as an unconfigured credential.
def _configured(value: str) -> bool:
    lower = value.strip().lower()
    return bool(lower) and not lower.startswith(("your-", "your_", "replace-", "replace_", "<"))


# Normalize the optional OpenRouter alias and return only nonsecret startup settings.
def load_configuration(model: str | None = None) -> dict:
    load_env()
    key = os.environ.get("API_KEY", "").strip() or os.environ.get("OPENROUTER_API_KEY", "").strip()
    base_url = os.environ.get("BASE_URL", "").strip() or DEFAULT_BASE_URL
    selected_model = (model or os.environ.get("AOS_MODEL", "")).strip() or DEFAULT_MODEL
    if key:
        os.environ["API_KEY"] = key
    os.environ["BASE_URL"] = base_url
    try:
        address = urlsplit(base_url)
        valid_url = (address.scheme in {"http", "https"} and bool(address.hostname)
                     and not address.username and not address.password)
    except ValueError:
        valid_url = False
    ready = _configured(key) and _configured(base_url) and valid_url
    reason = ("Add API_KEY to .env and restart to enable new collaborations." if not _configured(key)
              else "Set a valid BASE_URL in .env and restart to enable new collaborations.")
    return {"model": selected_model, "base_url": base_url, "live_enabled": bool(ready),
            "disabled_reason": "" if ready else reason}


# Check local credential presence without probing an API or spending tokens.
def credentials_ready() -> bool:
    return load_configuration()["live_enabled"]
