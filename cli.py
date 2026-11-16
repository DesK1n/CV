"""Unified Command-Line Interface (CLIApp) for CCTV Vehicle Tracking.

Dispatches high-level workflows:
1. prepare - Verifies or downloads the UA-DETRAC dataset and generates mock test videos.
2. train   - Launches model training (YOLO26, Mamba-YOLO, or full overnight suite).
3. track   - Runs Multi-Object Tracking (Enhanced SORT, ByteTrack, BoT-SORT) on video/sequences.
4. eval    - Benchmarks tracking accuracy (MOTA, IDF1, FPS, ID switches) on UA-DETRAC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

# Ensure Lab1 directory is on sys.path regardless of execution working directory
LAB1_DIR: Path = Path(__file__).resolve().parent
if str(LAB1_DIR) not in sys.path:
    sys.path.insert(0, str(LAB1_DIR))

import cv2

from config import (
    DATA_DIR,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_YOLO_WEIGHTS,
    DEVICE,
    MAMBA_YOLO_WEIGHTS,
    MODELS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    UA_DETRAC_DIR,
    UAV_YOLO_WEIGHTS,
)
from data_manager import UADetracDatasetManager, create_mock_video
from tracker import TrafficTracker


class CLIApp:
    """Central dispatcher for all CCTV vehicle tracking commands."""

    def __init__(self) -> None:
        """Initializes the CLI application parser and command mappings."""
        self.parser: argparse.ArgumentParser = self.build_parser()

    def build_parser(self) -> argparse.ArgumentParser:
        """Constructs argument parser hierarchy with subcommands.

        Returns:
            argparse.ArgumentParser: Root parser configured with subcommands.
        """
        parser = argparse.ArgumentParser(
            prog="traffic-cli",
            description="CCTV Vehicle Detection and Multi-Object Tracking Pipeline CLI.",
        )
        subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

        # 1. prepare
        prep_parser = subparsers.add_parser(
            "prepare", help="Verify/download UA-DETRAC dataset or create mock video."
        )
        prep_parser.add_argument(
            "--dest",
            type=Path,
            default=UA_DETRAC_DIR,
            help="Path to UA-DETRAC dataset directory.",
        )
        prep_parser.add_argument(
            "--verify-only",
            action="store_true",
            help="Only inspect and verify existing dataset without downloading.",
        )
        prep_parser.add_argument(
            "--create-mock-video",
            type=Path,
            default=None,
            help="Generate a synthetic MP4 test video at specified path.",
        )
        prep_parser.add_argument(
            "--frames",
            type=int,
            default=60,
            help="Number of frames for mock video generation.",
        )

        # 2. train
        train_parser = subparsers.add_parser(
            "train", help="Train vehicle detectors on UA-DETRAC."
        )
        train_parser.add_argument(
            "--model",
            choices=["yolo", "mamba", "all"],
            default="all",
            help="Model architecture to train.",
        )
        train_parser.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="Number of training epochs (defaults to standard hyperparameter).",
        )
        train_parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Mini-batch size for training.",
        )

        # 3. track
        track_parser = subparsers.add_parser(
            "track", help="Run multi-object vehicle tracking on video or sequence."
        )
        track_parser.add_argument(
            "--source",
            type=str,
            required=True,
            help="Path to input video file (.mp4, .mkv), image sequence dir, or webcam ID.",
        )
        track_parser.add_argument(
            "--tracker",
            choices=["enhanced_sort", "bytetrack", "botsort"],
            default="enhanced_sort",
            help="Tracking algorithm to execute.",
        )
        track_parser.add_argument(
            "--model",
            type=str,
            default="yolo26",
            help="Detector model path or identifier (yolo26, uav_yolo26s, mamba_yolo).",
        )
        track_parser.add_argument(
            "--conf",
            type=float,
            default=DEFAULT_CONF_THRESHOLD,
            help="Confidence threshold for vehicle detections.",
        )
        track_parser.add_argument(
            "--save-video",
            action="store_true",
            help="Save rendered output video with bounding boxes and trails.",
        )
        track_parser.add_argument(
            "--out-video",
            type=Path,
            default=None,
            help="Output video destination path.",
        )
        track_parser.add_argument(
            "--out-json",
            type=Path,
            default=None,
            help="Output JSON file for per-frame tracking telemetry.",
        )
        track_parser.add_argument(
            "--no-trails",
            action="store_true",
            help="Disable motion trajectory trail drawing.",
        )
        track_parser.add_argument(
            "--max-frames",
            type=int,
            default=None,
            help="Maximum frames to process.",
        )

        # 4. eval
        eval_parser = subparsers.add_parser(
            "eval", help="Evaluate tracking algorithms on UA-DETRAC validation set."
        )
        eval_parser.add_argument(
            "--sequence",
            type=str,
            default="MVI_20012",
            help="UA-DETRAC sequence ID prefix to benchmark.",
        )
        eval_parser.add_argument(
            "--split",
            choices=["val", "train"],
            default="val",
            help="Dataset split to evaluate.",
        )
        eval_parser.add_argument(
            "--max-frames",
            type=int,
            default=50,
            help="Number of frames per sequence to evaluate.",
        )

        return parser

    def run(self, args: Optional[Sequence[str]] = None) -> int:
        """Parses arguments and dispatches to appropriate handler.

        Args:
            args: Optional command line argument list. If None, uses sys.argv[1:].

        Returns:
            int: Process exit code (0 for success, non-zero for error).
        """
        parsed_args = self.parser.parse_args(args)
        if parsed_args.command is None:
            self.parser.print_help()
            return 0

        handler_name = f"handle_{parsed_args.command}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            print(f"[ERROR] Unknown subcommand: {parsed_args.command}", file=sys.stderr)
            return 1

        try:
            return handler(parsed_args)
        except Exception as exc:
            print(f"[ERROR] Execution failed: {exc}", file=sys.stderr)
            return 1

    def handle_prepare(self, args: argparse.Namespace) -> int:
        """Handles the 'prepare' subcommand.

        Args:
            args: Parsed arguments namespace.

        Returns:
            int: Exit code.
        """
        if args.create_mock_video:
            out_vid = create_mock_video(
                output_path=args.create_mock_video,
                num_frames=args.frames,
            )
            print(f"[SUCCESS] Mock test video created at: {out_vid}")
            return 0

        manager = UADetracDatasetManager(destination_dir=args.dest)
        if args.verify_only:
            valid = manager.is_valid()
            stats = manager.verify_dataset_structure()
            print(f"[INFO] UA-DETRAC valid: {valid}")
            for k, v in stats.items():
                print(f"  - {k}: {v}")
            return 0 if valid else 1

        print("[INFO] Preparing UA-DETRAC dataset...")
        dest = manager.prepare_dataset(download_if_missing=True)
        print(f"[SUCCESS] Dataset verified and ready at: {dest}")
        return 0

    def handle_train(self, args: argparse.Namespace) -> int:
        """Handles the 'train' subcommand.

        Args:
            args: Parsed arguments namespace.

        Returns:
            int: Exit code.
        """
        if args.model == "yolo":
            from train_yolo import train_yolo26

            print("[INFO] Launching YOLO26 UA-DETRAC training...")
            epochs = args.epochs if args.epochs is not None else 100
            batch = args.batch_size if args.batch_size is not None else 32
            train_yolo26(epochs=epochs, batch_size=batch)
        elif args.model == "mamba":
            from train_mamba import train_mamba_yolo

            print("[INFO] Launching Mamba-YOLO UA-DETRAC training...")
            epochs = args.epochs if args.epochs is not None else 30
            batch = args.batch_size if args.batch_size is not None else 16
            train_mamba_yolo(epochs=epochs, batch_size=batch)
        else:
            from train_all import run_overnight_pipeline

            print("[INFO] Launching complete end-to-end training pipeline...")
            run_overnight_pipeline()

        print("[SUCCESS] Training execution completed.")
        return 0

    def handle_track(self, args: argparse.Namespace) -> int:
        """Handles the 'track' subcommand.

        Args:
            args: Parsed arguments namespace.

        Returns:
            int: Exit code.
        """
        print(f"[INFO] Initializing TrafficTracker with tracker={args.tracker}, model={args.model}...")
        tracker = TrafficTracker(
            model_path=args.model,
            tracker_type=args.tracker,
            conf_thresh=args.conf,
            draw_trails=not args.no_trails,
        )

        source_path = Path(args.source)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        video_writer: Optional[cv2.VideoWriter] = None
        if args.save_video or args.out_video:
            out_vid_path = (
                args.out_video
                if args.out_video is not None
                else OUTPUTS_DIR / f"{source_path.stem}_{args.tracker}_tracked.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Probe frame dimensions from source
            cap_probe = cv2.VideoCapture(str(source_path))
            pw = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            ph = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            pfps = cap_probe.get(cv2.CAP_PROP_FPS) or 25.0
            cap_probe.release()
            video_writer = cv2.VideoWriter(str(out_vid_path), fourcc, pfps, (pw, ph))
            print(f"[INFO] Saving tracked video to: {out_vid_path}")

        records: List[Dict[str, Any]] = []
        frame_idx = 0

        for frame_annotated, stats in tracker.stream_frames(
            source=args.source,
            max_frames=args.max_frames,
            draw=True,
            yield_jpeg=False,
        ):
            records.append(stats)
            if video_writer is not None and isinstance(frame_annotated, np.ndarray):
                video_writer.write(frame_annotated)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(
                    f"  [Frame {frame_idx:04d}] FPS: {stats['fps']:.1f} | "
                    f"Active: {stats['active_tracks']} | Total: {stats['total_count']}"
                )

        if video_writer is not None:
            video_writer.release()
            print(f"[SUCCESS] Video rendered and saved.")

        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            print(f"[SUCCESS] Telemetry log saved to: {args.out_json}")

        print(
            f"[SUCCESS] Tracking complete. Processed {frame_idx} frames. "
            f"Unique vehicles identified: {tracker.unique_track_ids} (total: {len(tracker.unique_track_ids)})."
        )
        return 0

    def handle_eval(self, args: argparse.Namespace) -> int:
        """Handles the 'eval' subcommand.

        Args:
            args: Parsed arguments namespace.

        Returns:
            int: Exit code.
        """
        from evaluate import run_benchmark, save_reports

        print(f"[INFO] Running evaluation benchmark on sequence '{args.sequence}'...")
        results = run_benchmark(
            sequence_prefix=args.sequence,
            split=args.split,
            max_frames=args.max_frames,
        )
        save_reports(results)
        print("[SUCCESS] Evaluation benchmark finished and reports updated.")
        return 0


def main(args: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        args: Command line arguments.

    Returns:
        int: Process exit code.
    """
    app = CLIApp()
    return app.run(args)


if __name__ == "__main__":
    sys.exit(main())
