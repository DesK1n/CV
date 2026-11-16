"""FastAPI + WebSocket Application for UA-DETRAC Traffic Tracking Dashboard.

Provides REST endpoints for querying available videos, detection models,
and tracking algorithms, along with a full-duplex WebSocket stream endpoint
delivering real-time vehicle tracking frames and live MOT telemetry.
"""
from __future__ import annotations

import asyncio
import base64
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Register parent directory to import lab modules
LAB_DIR: Path = Path(__file__).resolve().parent.parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from config import (
    BASE_DIR,
    DATA_DIR,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_YOLO_WEIGHTS,
    MAX_AGE,
    MODELS_DIR,
    UAV_YOLO_WEIGHTS,
    WEB_DIR,
)
from tracker import TrafficTracker

STATIC_DIR: Path = Path(__file__).resolve().parent / "static"

app: FastAPI = FastAPI(
    title="UA-DETRAC Traffic Tracking Dashboard",
    description=(
        "Real-time multi-object tracking platform with YOLO26 and WebSocket streaming."
    ),
    version="1.0.0",
)

# Mount static asset directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index() -> FileResponse:
    """Serves the primary web dashboard HTML document.

    Returns:
        FileResponse: The index.html dashboard file.

    Raises:
        HTTPException: If index.html is missing.
    """
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(index_file)


@app.get("/api/videos")
async def get_videos() -> List[Dict[str, str]]:
    """Retrieves list of accessible CCTV and UAV video streams.

    Returns:
        List[Dict[str, str]]: Available video records with id, name, and size.
    """
    available_videos: List[Dict[str, str]] = []
    raw_videos = TrafficTracker.get_available_videos()
    for v in raw_videos:
        available_videos.append(
            {
                "id": v["name"],
                "name": f"{v['name']} ({v.get('size_mb', 'N/A')})",
                "path": v["path"],
                "size_mb": v.get("size_mb", "N/A"),
            }
        )
    return available_videos


@app.get("/api/models")
async def get_models() -> List[Dict[str, Any]]:
    """Retrieves list of supported YOLO26 vehicle detection models.

    Returns:
        List[Dict[str, Any]]: Models with identifiers, names, and default status.
    """
    models: List[Dict[str, Any]] = [
        {
            "id": "yolo26_uadetrac",
            "name": "YOLO26n UA-DETRAC CCTV (Default)",
            "path": str(DEFAULT_YOLO_WEIGHTS),
            "is_default": True,
            "exists": DEFAULT_YOLO_WEIGHTS.exists(),
        },
        {
            "id": "uav_yolo26s",
            "name": "UAV YOLO26s Aerial Drone (Optional)",
            "path": str(UAV_YOLO_WEIGHTS),
            "is_default": False,
            "exists": UAV_YOLO_WEIGHTS.exists(),
        },
    ]
    return models


@app.get("/api/trackers")
async def get_trackers() -> List[Dict[str, str]]:
    """Retrieves list of supported Multi-Object Tracking algorithms.

    Returns:
        List[Dict[str, str]]: Tracker metadata with id and display name.
    """
    return TrafficTracker.get_available_trackers()


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket) -> None:
    """Manages full-duplex WebSocket session for real-time video tracking.

    Receives control commands (start, pause, resume, stop, update_config)
    and streams back JPEG frames bundled with MOT telemetry statistics.

    Args:
        websocket: The active WebSocket connection instance.
    """
    await websocket.accept()

    state: Dict[str, Any] = {
        "is_streaming": False,
        "is_paused": False,
        "video": None,
        "model": "yolo26_uadetrac",
        "tracker": "enhanced_sort",
        "conf_thresh": DEFAULT_CONF_THRESHOLD,
        "max_age": MAX_AGE,
        "show_trails": True,
    }

    active_tracker: Optional[TrafficTracker] = None
    stream_gen = None
    total_video_frames: int = 0

    def resolve_video_path(video_identifier: str) -> Optional[Path]:
        """Resolves video filename or path string into an existing Path.

        Args:
            video_identifier: File name or path string.

        Returns:
            Optional[Path]: Valid filesystem Path or None.
        """
        candidate = Path(video_identifier)
        if candidate.exists() and candidate.is_file():
            return candidate
        dataset_candidate = DATA_DIR / candidate.name
        if dataset_candidate.exists() and dataset_candidate.is_file():
            return dataset_candidate
        return None

    def initialize_stream() -> None:
        """Initializes TrafficTracker engine and frame generator."""
        nonlocal active_tracker, stream_gen, total_video_frames

        vid_path = resolve_video_path(state["video"])
        if vid_path is None:
            raise FileNotFoundError(f"Video '{state['video']}' could not be located.")

        cap = cv2.VideoCapture(str(vid_path))
        total_video_frames = (
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        )
        cap.release()

        active_tracker = TrafficTracker(
            model_path=state["model"],
            tracker_type=state["tracker"],
            conf_thresh=state["conf_thresh"],
            max_age=state["max_age"],
            draw_trails=state["show_trails"],
        )

        stream_gen = active_tracker.stream_frames(
            source=vid_path,
            loop=False,
            draw=True,
            yield_jpeg=True,
            jpeg_quality=75,
        )

    try:
        while True:
            # Poll client messages without blocking frame transmission
            try:
                msg_raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.015
                )
                import json

                msg = json.loads(msg_raw)
                action = msg.get("action")

                if action == "start":
                    state["video"] = msg.get("video")
                    state["model"] = msg.get("model", "yolo26_uadetrac")
                    state["tracker"] = msg.get("tracker", "enhanced_sort")
                    state["conf_thresh"] = float(
                        msg.get("conf_thresh", DEFAULT_CONF_THRESHOLD)
                    )
                    state["max_age"] = int(msg.get("max_age", MAX_AGE))
                    state["show_trails"] = bool(msg.get("show_trails", True))

                    try:
                        initialize_stream()
                        state["is_streaming"] = True
                        state["is_paused"] = False
                    except Exception as exc:
                        await websocket.send_json(
                            {
                                "type": "status",
                                "status": "error",
                                "message": str(exc),
                            }
                        )

                elif action == "pause":
                    state["is_paused"] = True

                elif action == "resume":
                    state["is_paused"] = False

                elif action == "stop":
                    state["is_streaming"] = False
                    state["is_paused"] = False
                    stream_gen = None
                    active_tracker = None

                elif action == "update_config":
                    if "conf_thresh" in msg:
                        state["conf_thresh"] = float(msg["conf_thresh"])
                        if active_tracker is not None:
                            active_tracker.conf_thresh = state["conf_thresh"]
                    if "max_age" in msg:
                        state["max_age"] = int(msg["max_age"])
                        if active_tracker is not None:
                            active_tracker.max_age = state["max_age"]
                    if "show_trails" in msg:
                        state["show_trails"] = bool(msg["show_trails"])
                        if active_tracker is not None:
                            active_tracker.draw_trails = state["show_trails"]

            except asyncio.TimeoutError:
                pass

            # If streaming and not paused, yield next frame
            should_stream = (
                state["is_streaming"]
                and not state["is_paused"]
                and stream_gen is not None
            )
            if should_stream:
                try:
                    # Run frame processing in thread pool to prevent blocking loop
                    frame_jpeg, stats = await asyncio.to_thread(next, stream_gen)
                    b64_frame = base64.b64encode(frame_jpeg).decode("ascii")

                    # Compute live class distribution
                    classes_counter: Counter[str] = Counter()
                    for trk in stats.get("tracks", []):
                        raw_cls = str(trk.get("cls", "car")).lower()
                        normalized_cls = (
                            "car" if raw_cls in ("vehicle", "auto") else raw_cls
                        )
                        classes_counter[normalized_cls] += 1

                    class_counts: Dict[str, int] = {
                        "car": classes_counter["car"],
                        "bus": classes_counter["bus"],
                        "van": classes_counter["van"],
                        "truck": classes_counter["truck"],
                    }

                    payload: Dict[str, Any] = {
                        "type": "frame",
                        "frame": b64_frame,
                        "fps": stats.get("fps", 0.0),
                        "active_tracks": stats.get("active_tracks", 0),
                        "total_tracks": stats.get("total_count", 0),
                        "frame_idx": stats.get("frame_idx", 0),
                        "total_frames": total_video_frames,
                        "classes": class_counts,
                    }
                    await websocket.send_json(payload)

                except StopIteration:
                    # Video stream completed
                    state["is_streaming"] = False
                    stream_gen = None
                    await websocket.send_json(
                        {
                            "type": "status",
                            "status": "completed",
                            "message": "Stream finished",
                        }
                    )
                except Exception as stream_err:
                    state["is_streaming"] = False
                    stream_gen = None
                    await websocket.send_json(
                        {
                            "type": "status",
                            "status": "error",
                            "message": f"Streaming error: {stream_err}",
                        }
                    )

            await asyncio.sleep(0.005)

    except WebSocketDisconnect:
        # Client gracefully or abruptly disconnected
        state["is_streaming"] = False
        stream_gen = None
        active_tracker = None


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launches the Uvicorn ASGI server instance.

    Args:
        host: Network interface address.
        port: TCP network port.
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
