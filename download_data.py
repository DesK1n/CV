"""Downloader and manager for UA-DETRAC CCTV traffic dataset.

Re-exports and wraps UADetracDatasetManager from data_manager for backward compatibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config import UA_DETRAC_DIR
from data_manager import KAGGLE_UADETRAC_REF, UADetracDatasetManager


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=UA_DETRAC_DIR, help="Destination directory")
    args = parser.parse_args()

    manager = UADetracDatasetManager(destination_dir=args.dest)
    manager.prepare_dataset(download_if_missing=True)


if __name__ == "__main__":
    main()
