"""Environment-aware API key protection for APEX inference endpoints."""
from __future__ import annotations

import hmac
import os

from dotenv import load_dotenv

load_dotenv()

PUBLIC_PATHS = {"/", "/health", "/health/live", "/health/ready", "/docs", "/openapi.json", "/redoc"}


def security_settings() -> tuple[str, str]:
    environment = os.getenv("APEX_ENV", "development").strip().lower()
    api_key = os.getenv("APEX_API_KEY", "").strip()
    if environment not in {"development", "test", "production"}:
        raise RuntimeError("APEX_ENV must be development, test, or production")
    if api_key and len(api_key) < 32:
        raise RuntimeError("APEX_API_KEY must contain at least 32 characters")
    if environment == "production" and not api_key:
        raise RuntimeError("APEX_API_KEY is required when APEX_ENV=production")
    return environment, api_key


def api_key_matches(provided: str | None, expected: str) -> bool:
    return bool(provided and expected and hmac.compare_digest(provided, expected))
