#!/usr/bin/env python3
"""
app/gradio_app.py  ·  CLI entrypoint delegating to app.app.
===========================================================
Usage:
    python app/gradio_app.py --checkpoint outputs/dinov2_frozen/best_model.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import main

if __name__ == "__main__":
    main()
