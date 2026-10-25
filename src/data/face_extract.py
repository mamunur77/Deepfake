"""
src.data.face_extract  ·  MTCNN-based face detection & cropping.
================================================================
Extracts faces from videos and images using facenet-pytorch's MTCNN.

Pipeline (per video):
    1. Sample N evenly-spaced frames with OpenCV.
    2. Detect the largest face in each frame (MTCNN, select_largest=True).
    3. Expand the bounding box by `margin` % on each side.
    4. Crop and resize to `output_size × output_size`.
    5. Save as  {output_dir}/{label}/{item_id}/frame_XXXX.jpg.

For plain images the same detect → crop → resize → save logic applies,
producing one output per input image.

Skipping:
    If the output sub-directory already contains ≥1 file, the item is
    skipped (resume-friendly).

No-face logging:
    Items where zero faces were detected are collected and written to
    {output_dir}/no_face_detected.log.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

__all__ = [
    "create_mtcnn",
    "extract_frames_from_video",
    "detect_and_crop_face",
    "process_video",
    "process_image",
    "run_extraction",
]

logger = logging.getLogger(__name__)

# ── Supported file extensions ────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ═════════════════════════════════════════════════════════════════
#  MTCNN initialisation
# ═════════════════════════════════════════════════════════════════

def create_mtcnn(device: str = "cpu") -> MTCNN:
    """
    Build an MTCNN detector that returns only the largest face.

    Parameters
    ----------
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    MTCNN
    """
    return MTCNN(
        image_size=160,       # internal detection size
        margin=0,             # we add our own margin later
        keep_all=False,       # one face per image
        select_largest=True,  # pick the biggest bounding box
        post_process=False,   # return raw PIL crop, not normalised
        device=device,
    )


# ═════════════════════════════════════════════════════════════════
#  Frame extraction from video
# ═════════════════════════════════════════════════════════════════

def extract_frames_from_video(
    video_path: str | Path,
    n_frames: int = 16,
) -> list[tuple[int, Image.Image]]:
    """
    Read *n_frames* evenly-spaced frames from *video_path*.

    Parameters
    ----------
    video_path : str | Path
        Path to the video file.
    n_frames : int
        How many frames to sample.

    Returns
    -------
    list[tuple[int, PIL.Image.Image]]
        List of ``(frame_index, PIL RGB image)`` pairs.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        logger.warning("Zero frames reported for: %s", video_path)
        cap.release()
        return []

    # Clamp n_frames so we never ask for more than total
    n_frames = min(n_frames, total)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    frames: list[tuple[int, Image.Image]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append((int(idx), Image.fromarray(rgb)))
        else:
            logger.debug("Failed to read frame %d in %s", idx, video_path)

    cap.release()
    return frames


# ═════════════════════════════════════════════════════════════════
#  Face detection + cropping
# ═════════════════════════════════════════════════════════════════

def detect_and_crop_face(
    image: Image.Image,
    mtcnn: MTCNN,
    margin_pct: int = 20,
    output_size: int = 224,
) -> Image.Image | None:
    """
    Detect the largest face, expand the box by *margin_pct* %, crop & resize.

    Parameters
    ----------
    image : PIL.Image.Image
        Input RGB image.
    mtcnn : MTCNN
        Pre-initialised face detector.
    margin_pct : int
        Percentage of the bbox width/height to add on each side.
    output_size : int
        Final square side length in pixels.

    Returns
    -------
    PIL.Image.Image | None
        Cropped face resized to ``output_size × output_size``,
        or ``None`` when no face is found.
    """
    boxes, _probs = mtcnn.detect(image)

    if boxes is None or len(boxes) == 0:
        return None

    # boxes[0] is the largest face (select_largest=True)
    x1, y1, x2, y2 = (int(b) for b in boxes[0])

    # --- expand by margin_pct % on each side ---
    w, h = x2 - x1, y2 - y1
    mx = int(w * margin_pct / 100)
    my = int(h * margin_pct / 100)
    img_w, img_h = image.size

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(img_w, x2 + mx)
    y2 = min(img_h, y2 + my)

    face = image.crop((x1, y1, x2, y2))
    face = face.resize((output_size, output_size), Image.LANCZOS)
    return face


# ═════════════════════════════════════════════════════════════════
#  Per-item processors
# ═════════════════════════════════════════════════════════════════

def process_video(
    video_path: Path,
    output_dir: Path,
    mtcnn: MTCNN,
    n_frames: int = 16,
    margin: int = 20,
    size: int = 224,
) -> tuple[int, int]:
    """
    Extract faces from one video and save crops.

    Returns
    -------
    tuple[int, int]
        ``(faces_saved, frames_with_no_face)``.
    """
    frames = extract_frames_from_video(video_path, n_frames)
    if not frames:
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    no_face = 0

    for frame_idx, pil_img in frames:
        face = detect_and_crop_face(pil_img, mtcnn, margin, size)
        if face is not None:
            save_path = output_dir / f"frame_{frame_idx:04d}.jpg"
            face.save(save_path, quality=95)
            saved += 1
        else:
            no_face += 1

    return saved, no_face


def process_image(
    image_path: Path,
    output_dir: Path,
    mtcnn: MTCNN,
    margin: int = 20,
    size: int = 224,
) -> bool:
    """
    Detect the largest face in a single image, crop & save.

    Returns
    -------
    bool
        ``True`` if a face was saved.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.warning("Cannot open image %s: %s", image_path, exc)
        return False

    face = detect_and_crop_face(image, mtcnn, margin, size)
    if face is None:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{image_path.stem}.jpg"
    face.save(save_path, quality=95)
    return True


# ═════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════

def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _is_already_done(out_dir: Path) -> bool:
    """Return True if *out_dir* exists and contains at least one file."""
    if not out_dir.exists():
        return False
    return any(out_dir.iterdir())


def _collect_items(input_dir: Path) -> list[tuple[Path, str, str]]:
    """
    Walk ``input_dir/{real,fake}/`` and return a sorted list of
    ``(file_path, label, item_id)`` tuples for every video or image.
    """
    items: list[tuple[Path, str, str]] = []
    for label in ("real", "fake"):
        label_dir = input_dir / label
        if not label_dir.exists():
            logger.warning("Label directory not found: %s", label_dir)
            continue
        for fp in sorted(label_dir.iterdir()):
            if fp.is_file() and (_is_video(fp) or _is_image(fp)):
                items.append((fp, label, fp.stem))
    return items


# ═════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═════════════════════════════════════════════════════════════════

def run_extraction(
    input_dir: str | Path,
    output_dir: str | Path,
    n_frames: int = 16,
    limit: int | None = None,
    margin: int = 20,
    size: int = 224,
    device: str = "cpu",
) -> dict:
    """
    End-to-end extraction pipeline.

    Parameters
    ----------
    input_dir : path-like
        Root that must contain ``real/`` and/or ``fake/`` sub-directories.
    output_dir : path-like
        Where cropped faces are written.
    n_frames : int
        Frames per video (evenly spaced).
    limit : int | None
        Process at most this many items (``None`` = all).
    margin : int
        Bounding-box expansion in percent.
    size : int
        Output face crop side-length.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    dict
        Extraction statistics.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── gather files ──────────────────────────────────────────
    all_items = _collect_items(input_dir)
    total_found = len(all_items)

    if total_found == 0:
        print(f"[extract] No videos or images found in {input_dir}/{{real,fake}}/")
        return {"total_items": 0, "processed": 0, "skipped": 0,
                "faces_saved": 0, "no_face": []}

    # ── apply limit ───────────────────────────────────────────
    if limit is not None and limit < total_found:
        print(f"[extract] Limiting to {limit} / {total_found} items")
        all_items = all_items[:limit]

    effective_count = len(all_items)

    # ── init MTCNN ────────────────────────────────────────────
    print(f"[extract] Initialising MTCNN on {device} ...")
    mtcnn = create_mtcnn(device)
    print(f"[extract] Processing {effective_count} items from {input_dir}")

    # ── main loop ─────────────────────────────────────────────
    stats = {"total_items": total_found, "processed": 0,
             "skipped": 0, "faces_saved": 0, "no_face": []}

    for file_path, label, item_id in tqdm(all_items, desc="Extracting faces"):
        out_subdir = output_dir / label / item_id

        # Skip already-processed items
        if _is_already_done(out_subdir):
            stats["skipped"] += 1
            continue

        if _is_video(file_path):
            saved, _missed = process_video(
                file_path, out_subdir, mtcnn, n_frames, margin, size,
            )
            if saved == 0:
                stats["no_face"].append(str(file_path))
                logger.warning("No faces in video: %s", file_path)
            stats["faces_saved"] += saved

        elif _is_image(file_path):
            ok = process_image(file_path, out_subdir, mtcnn, margin, size)
            if not ok:
                stats["no_face"].append(str(file_path))
                logger.warning("No face in image: %s", file_path)
            else:
                stats["faces_saved"] += 1

        stats["processed"] += 1

    # ── write no-face log ─────────────────────────────────────
    if stats["no_face"]:
        log_path = output_dir / "no_face_detected.log"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n--- Run: {datetime.now().isoformat()} ---\n")
            for entry in stats["no_face"]:
                fh.write(f"{entry}\n")
        print(f"[extract] {len(stats['no_face'])} items with no face "
              f"-> {log_path}")

    # ── summary ───────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Extraction Summary")
    print(f"{'=' * 55}")
    print(f"  Total items found : {stats['total_items']}")
    print(f"  Processed         : {stats['processed']}")
    print(f"  Skipped (exists)  : {stats['skipped']}")
    print(f"  Faces saved       : {stats['faces_saved']}")
    print(f"  No face detected  : {len(stats['no_face'])}")
    print(f"  Output directory  : {output_dir}")
    print(f"{'=' * 55}")

    return stats
