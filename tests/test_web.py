"""Unit tests for the FastAPI web dashboard in web/app.py.

Tests REST API endpoints (/api/videos, /api/models, /api/trackers, /)
and WebSocket communication endpoint (/ws/stream).
"""
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

# Add Lab1 and web root to sys.path
LAB1_ROOT = Path(__file__).resolve().parent.parent
if str(LAB1_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB1_ROOT))

from web.app import app


@pytest.fixture
def client() -> TestClient:
    """Fixture providing FastAPI test client.

    Returns:
        TestClient: Synchronous test client instance.
    """
    return TestClient(app)


def test_serve_index(client: TestClient) -> None:
    """Verify that root endpoint serves the primary index.html dashboard.

    Args:
        client: FastAPI test client fixture.

    Returns:
        None
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "UA-DETRAC" in response.text or "Vehicle Tracking" in response.text


def test_api_videos(client: TestClient) -> None:
    """Verify GET /api/videos endpoint returns list of video objects.

    Args:
        client: FastAPI test client fixture.

    Returns:
        None
    """
    response = client.get("/api/videos")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    for item in data:
        assert "id" in item
        assert "name" in item
        assert "path" in item


def test_api_models(client: TestClient) -> None:
    """Verify GET /api/models returns model records with default YOLO26n.

    Args:
        client: FastAPI test client fixture.

    Returns:
        None
    """
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    model_ids = [m["id"] for m in data]
    assert "yolo26_uadetrac" in model_ids

    default_model = next(m for m in data if m["id"] == "yolo26_uadetrac")
    assert default_model.get("is_default") is True


def test_api_trackers(client: TestClient) -> None:
    """Verify GET /api/trackers returns list of supported tracking algorithms.

    Args:
        client: FastAPI test client fixture.

    Returns:
        None
    """
    response = client.get("/api/trackers")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    tracker_ids = [t["id"] for t in data]
    assert "enhanced_sort" in tracker_ids
    assert "bytetrack" in tracker_ids
    assert "botsort" in tracker_ids


def test_websocket_stream_connection(client: TestClient) -> None:
    """Verify WebSocket /ws/stream accepts connection and handles command/disconnect.

    Args:
        client: FastAPI test client fixture.

    Returns:
        None
    """
    with client.websocket_connect("/ws/stream") as websocket:
        # Send an update_config command
        websocket.send_json(
            {
                "command": "update_config",
                "conf_thresh": 0.35,
                "max_age": 20,
                "show_trails": False,
            }
        )
        # Send stop command
        websocket.send_json({"command": "stop"})
