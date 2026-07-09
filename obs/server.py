"""FastAPI app exposing ``/metrics`` for Prometheus scrape (OBS-01).

Wires a single ``MetricsRegistry`` instance to the FastAPI app; Grafana
or a sidecar scraper hits ``GET /metrics`` to read the text exposition
format.

Usage:

    from obs.registry import MetricsRegistry
    from obs.server import build_app

    registry = MetricsRegistry()
    app = build_app(registry)
    # uvicorn obs.server:app --port 9100

In tests we use ``httpx.ASGITransport`` against ``build_app`` — no
network needed.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from obs.registry import MetricsRegistry


def build_app(registry: MetricsRegistry | None = None) -> FastAPI:
    """Return a FastAPI app exposing ``/metrics`` for the given registry."""
    if registry is None:
        registry = MetricsRegistry()
    app = FastAPI(title="goodputlab-obs", version="0.1.0")
    app.state.registry = registry

    @app.get("/metrics")
    def metrics() -> Response:
        body = generate_latest(registry.registry)
        return Response(content=body, media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


# Module-level app for uvicorn entrypoint.
app = build_app()


__all__ = ["build_app"]