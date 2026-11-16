"""Pytest configuration and common fixtures."""
import sys
from pathlib import Path

# Add Lab1 root to sys.path
LAB1_ROOT = Path(__file__).resolve().parent.parent
if str(LAB1_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB1_ROOT))
