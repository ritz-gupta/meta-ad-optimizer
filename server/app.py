"""
FastAPI application for the Meta Ad Optimizer Environment.

Exposes the AdOptimizerEnvironment over HTTP and WebSocket
endpoints using the standard OpenEnv ``create_app`` factory.

Usage:
    uvicorn meta_ad_optimizer.server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import AdAction, AdObservation
    from .ad_environment import AdOptimizerEnvironment
except ImportError:
    from models import AdAction, AdObservation
    from server.ad_environment import AdOptimizerEnvironment

app = create_app(
    AdOptimizerEnvironment,
    AdAction,
    AdObservation,
    env_name="meta_ad_optimizer",
    max_concurrent_envs=50,
)


@app.get("/")
def root():
    return {
        "name": "meta_ad_optimizer",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state", "/health"],
        "tasks": ["creative_matcher", "placement_optimizer", "campaign_optimizer"],
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for ``uv run server`` or ``python -m meta_ad_optimizer.server.app``."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
