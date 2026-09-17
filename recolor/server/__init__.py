"""HTTP layer: the FastAPI application in `recolor.server.app` (see docs/ARCHITECTURE.md §3.7)."""
from .app import create_app

__all__ = ["create_app"]
