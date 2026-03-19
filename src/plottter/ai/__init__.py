"""Optional AI image processing via Replicate.com.

All features in this package are optional.  Configure a Replicate API key in
Preferences > AI Integration to enable them.

Check availability before calling any API::

    from plottter.ai.replicate_client import ReplicateClient
    client = ReplicateClient(api_key)
    if client.is_available():
        ...
"""

try:
    from plottter.ai.replicate_client import ReplicateClient, ReplicateAPIError
except Exception:
    # If replicate_client itself fails to import (e.g. due to a broken pydantic
    # shim on Python 3.14+), expose safe fallback sentinels so that
    # ``from plottter.ai import ReplicateClient`` never raises at import time.
    ReplicateClient = None  # type: ignore[assignment,misc]
    ReplicateAPIError = Exception  # type: ignore[assignment,misc]

__all__ = ["ReplicateClient", "ReplicateAPIError"]
