"""
Analyze Inoc_IV pea leaf images for morphometry and Ascochyta symptoms.

Run from the repository root, or from RStudio with:

    reticulate::py_run_file("py/analyze_inoc_iv.py")

The script expects this layout:

    Inoc_IV/
      InocIV_inventario.xlsx
      R1/R1_0Datacolor_3dpi.jpg
      R1/R1_A_3dpi.jpg
      ...

Outputs are written to:

    Inoc_IV/results_inoc_iv/
      inoc_iv_leaf_measurements.csv
      inoc_iv_leaf_progression.csv
      inoc_iv_leaf_audpc.csv
      visual_checks/*.jpg

Measurements are in pixels. The stable leaf key is:
replicate + plate + grid position, joined with inventory accession metadata.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "Inoc_IV"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "results_inoc_iv"
INVENTORY_NAME = "InocIV_inventario.xlsx"

IMAGE_PATTERN = re.compile(r"^(R\d+)_(.+)_(\d+)dpi$", re.IGNORECASE)
DATACOLOR_CODE = "0Datacolor"

# Tunable defaults. Keep these close to the top for easy RStudio editing.
VISUAL_CHECKS = True
MAX_VISUAL_CHECKS = 30  # 0 means save checks for every analyzed image.
MIN_LEAF_AREA_FRAC = 0.0012
MAX_LEAF_AREA_FRAC = 0.0900
EXPECTED_LEAVES = 9

# === VISUAL THRESHOLD PLACEHOLDERS ==========================================
# Edit these values while comparing the generated visual_checks overlays.
# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255.
# OpenCV LAB ranges: L/a*/b* 0-255, with neutral a* and b* near 128.

# Leaf isolation from LAB b* channel.
LEAF_BSTAR_MIN = 132      # PLACEHOLDER: lower this if leaf edges disappear.
LEAF_S_MIN = 22           # PLACEHOLDER: raise this if labels/reflections enter.
LEAF_V_MIN = 35           # PLACEHOLDER: lower this for dark leaves/necrosis.
LEAF_V_MAX = 245          # PLACEHOLDER: lower this if bright glare is included.
LEAF_MEDIAN_BLUR = 5      # PLACEHOLDER: odd number, usually 3/5/7.
LEAF_OPEN_KERNEL = 5      # PLACEHOLDER: removes small noise.
LEAF_CLOSE_KERNEL = 11    # PLACEHOLDER: fills small gaps in leaf masks.

# Petri dish detection. Used only to estimate the fixed dish rectangle, unless
# --per-image-dish is used. You can bypass this with --dish x,y,w,h.
DISH_BRIGHT_THRESHOLDS = (90, 110, 130, 150, 170)  # PLACEHOLDER

# Pixel classification inside the isolated leaf mask.
HEALTHY_H_MIN = 38        # PLACEHOLDER
HEALTHY_H_MAX = 96        # PLACEHOLDER
HEALTHY_S_MIN = 32        # PLACEHOLDER
HEALTHY_V_MIN = 45        # PLACEHOLDER
HEALTHY_EXG_MIN = -10     # PLACEHOLDER: excess green lower bound.

CHLOROSIS_H_MIN = 18      # PLACEHOLDER
CHLOROSIS_H_MAX = 38      # PLACEHOLDER: exclusive upper bound.
CHLOROSIS_S_MIN = 25      # PLACEHOLDER
CHLOROSIS_V_MIN = 65      # PLACEHOLDER

NECROSIS_H_LOW_MAX = 24   # PLACEHOLDER
NECROSIS_H_HIGH_MIN = 165 # PLACEHOLDER
NECROSIS_S_MIN = 25       # PLACEHOLDER
NECROSIS_V_MAX = 170      # PLACEHOLDER
NECROSIS_A_MIN = 124      # PLACEHOLDER
NECROSIS_DARK_V_MAX = 70  # PLACEHOLDER
NECROSIS_DARK_S_MIN = 12  # PLACEHOLDER

DETERIORATION_S_MAX = 45  # PLACEHOLDER
DETERIORATION_V_MAX = 75  # PLACEHOLDER
DETERIORATION_B_MAX = 125 # PLACEHOLDER
# ===========================================================================


@dataclass
class ParsedName:
    replicate: str
    plate_code: str
    dpi: int
    is_datacolor: bool


@dataclass
class Calibration:
    replicate: str
    dpi: int
    gains_bgr: np.ndarray
    white_patch_bgr: Tuple[float, float, float]
    method: str


@dataclass
class Dish:
    x: int
    y: int
    w: int
    h: int
    confidence: str


@dataclass
class LeafObject:
    mask: np.ndarray
    contour: np.ndarray
    area_px: int
    centroid_x: float
    centroid_y: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Inoc_IV square Petri dish pea leaf images."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--inventory", default=None)
    parser.add_argument("--no-visual-checks", action="store_true")
    parser.add_argument(
        "--max-visual-checks",
        type=int,
        default=MAX_VISUAL_CHECKS,
        help="Maximum overlay images to save. Use 0 for every image.",
    )
    parser.add_argument(
        "--match",
        default="",
        help="Optional case-insensitive filename filter, e.g. R2_A or 8dpi.",
    )
    parser.add_argument(
        "--dish",
        default="",
        help="Optional fixed Petri dish rectangle as x,y,w,h pixels.",
    )
    parser.add_argument(
        "--per-image-dish",
        action="store_true",
        help="Detect the Petri dish separately for each image.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug: process only N images.")
    return parser.parse_args()


def parse_image_name(path: Path) -> Optional[ParsedName]:
    match = IMAGE_PATTERN.match(path.stem)
    if not match:
        return None
    replicate, plate_code, dpi = match.groups()
    return ParsedName(
        replicate=replicate.upper(),
        plate_code=plate_code,
        dpi=int(dpi),
        is_datacolor=plate_code.lower() == DATACOLOR_CODE.lower(),
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def col_to_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    value = 0
    for letter in letters:
        value = value * 26 + (ord(letter) - ord("A") + 1)
    return value - 1


def read_xlsx_sheets(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Read simple .xlsx sheets without openpyxl.

    This parser is intentionally small: it handles string and numeric cell values,
    which is enough for the InocIV inventory workbook.
    """
    if not path.exists():
        return {}

    ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    with zipfile.ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{ns_main}si"):
                text = "".join(node.text or "" for node in si.iter(f"{ns_main}t"))
                shared_strings.append(text)

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"].lstrip("/")
            for rel in rels.findall(f"{rel_ns}Relationship")
        }

        sheets: Dict[str, List[Dict[str, str]]] = {}
        for sheet in workbook.findall(f".//{ns_main}sheet"):
            sheet_name = sheet.attrib.get("name", "")
            rel_id = sheet.attrib.get(f"{ns_rel}id")
            if not rel_id or rel_id not in rel_map:
                continue
            target = rel_map[rel_id]
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            if sheet_path not in zf.namelist():
                continue

            root = ET.fromstring(zf.read(sheet_path))
            rows: List[List[str]] = []
            for row in root.findall(f".//{ns_main}row"):
                values: List[str] = []
                for cell in row.findall(f"{ns_main}c"):
                    idx = col_to_index(cell.attrib.get("r", "A"))
                    while len(values) <= idx:
                        values.append("")
                    node = cell.find(f"{ns_main}v")
                    value = "" if node is None or node.text is None else node.text
                    if cell.attrib.get("t") == "s" and value != "":
                        value = shared_strings[int(value)]
                    values[idx] = value
                rows.append(values)

            if not rows:
                sheets[sheet_name] = []
                continue

            headers = [h.strip() for h in rows[0]]
            records: List[Dict[str, str]] = []
            for row in rows[1:]:
                record = {
                    headers[i]: row[i].strip() if i < len(row) else ""
                    for i in range(len(headers))
                    if headers[i]
                }
                if any(record.values()):
                    records.append(record)
            sheets[sheet_name] = records

    return sheets


def load_inventory(inventory_path: Path) -> Dict[Tuple[str, str, int, int], str]:
    sheets = read_xlsx_sheets(inventory_path)
    inventory: Dict[Tuple[str, str, int, int], str] = {}

    for sheet_name, rows in sheets.items():
        if sheet_name.upper() not in {"R1", "R2", "R3"}:
            continue
        for row in rows:
            rep = row.get("Rep", sheet_name).strip().upper()
            plate = row.get("Plate", "").strip()
            if not rep or not plate:
                continue
            for key, accession in row.items():
                match = re.match(r"Accesion_pos_(\d):(\d)", key)
                if not match:
                    continue
                grid_row, grid_col = int(match.group(1)), int(match.group(2))
                accession = accession.strip()
                if accession and accession.upper() != "NA":
                    inventory[(rep, plate, grid_row, grid_col)] = accession

    return inventory


def read_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Could not read image: {path}")
    return img


def datacolor_white_patch(img_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
    """Estimate color gains from the brightest low-saturation chart patch.

    The Datacolor image has a dark frame plus regular color squares. We detect
    candidate patch blobs and choose the brightest neutral-looking one. If this
    fails, fall back to the brightest 1 percent of low-saturation pixels.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    min_area = max(200, int(h * w * 0.001))
    max_area = int(h * w * 0.035)

    candidate_mask = cv2.inRange(hsv, np.array([0, 0, 45]), np.array([179, 255, 255]))
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)
    )
    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1.0
    best_mean: Optional[np.ndarray] = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        ratio = bw / float(bh)
        if ratio < 0.65 or ratio > 1.45:
            continue
        patch_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(patch_mask, [contour], -1, 255, -1)
        sat = float(cv2.mean(hsv[:, :, 1], patch_mask)[0])
        val = float(cv2.mean(hsv[:, :, 2], patch_mask)[0])
        mean_bgr = np.array(cv2.mean(img_bgr, patch_mask)[:3], dtype=np.float32)
        channel_spread = float(np.max(mean_bgr) - np.min(mean_bgr))
        score = val - 1.8 * sat - 0.6 * channel_spread
        if score > best_score:
            best_score = score
            best_mean = mean_bgr

    if best_mean is not None and np.min(best_mean) > 20:
        gains = np.array([235.0, 235.0, 235.0], dtype=np.float32) / best_mean
        return np.clip(gains, 0.55, 1.85), "datacolor_white_patch"

    low_sat = hsv[:, :, 1] < 35
    bright = hsv[:, :, 2] > np.percentile(hsv[:, :, 2], 99)
    mask = (low_sat & bright).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return np.ones(3, dtype=np.float32), "none"
    mean_bgr = np.array(cv2.mean(img_bgr, mask)[:3], dtype=np.float32)
    gains = np.array([235.0, 235.0, 235.0], dtype=np.float32) / np.maximum(mean_bgr, 1.0)
    return np.clip(gains, 0.55, 1.85), "bright_neutral_pixels"


def build_calibrations(image_files: Sequence[Path]) -> Dict[Tuple[str, int], Calibration]:
    calibrations: Dict[Tuple[str, int], Calibration] = {}
    for path in image_files:
        parsed = parse_image_name(path)
        if not parsed or not parsed.is_datacolor:
            continue
        img = read_image(path)
        if img is None:
            continue
        gains, method = datacolor_white_patch(img)
        white_patch = tuple(float(v) for v in (np.array([235.0, 235.0, 235.0]) / gains))
        calibrations[(parsed.replicate, parsed.dpi)] = Calibration(
            replicate=parsed.replicate,
            dpi=parsed.dpi,
            gains_bgr=gains,
            white_patch_bgr=white_patch,
            method=method,
        )
    return calibrations


def apply_calibration(img_bgr: np.ndarray, calibration: Optional[Calibration]) -> np.ndarray:
    if calibration is None:
        return img_bgr.copy()
    calibrated = img_bgr.astype(np.float32) * calibration.gains_bgr.reshape(1, 1, 3)
    return np.clip(calibrated, 0, 255).astype(np.uint8)


def detect_leaf_mask(img_bgr: np.ndarray, dish: Optional[Dish] = None) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    b_lab = lab[:, :, 2]

    # In OpenCV LAB, b* is stored on a 0-255 scale centered near 128.
    # Pea leaf tissue is strongly positive in b* compared with labels, the
    # black mat, tape, and most Petri reflections.
    # THRESHOLD PLACEHOLDER: leaf/background segmentation.
    # Tune LEAF_* constants near the top of this script.
    bstar_leaf = (
        (b_lab >= LEAF_BSTAR_MIN)
        & (s >= LEAF_S_MIN)
        & (v >= LEAF_V_MIN)
        & (v <= LEAF_V_MAX)
    )
    mask = bstar_leaf.astype(np.uint8) * 255

    if dish is not None:
        dish_mask = np.zeros_like(mask)
        dish_mask[dish.y : dish.y + dish.h, dish.x : dish.x + dish.w] = 255
        mask = cv2.bitwise_and(mask, dish_mask)

    mask = cv2.medianBlur(mask, LEAF_MEDIAN_BLUR)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((LEAF_OPEN_KERNEL, LEAF_OPEN_KERNEL), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((LEAF_CLOSE_KERNEL, LEAF_CLOSE_KERNEL), np.uint8))
    return mask


def detect_dish(img_bgr: np.ndarray, leaf_mask: np.ndarray) -> Dish:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    best_threshold: Optional[Tuple[float, Tuple[int, int, int, int]]] = None
    upper = np.zeros_like(gray)
    upper[: int(h * 0.70), :] = 255
    # THRESHOLD PLACEHOLDER: Petri dish brightness sweep.
    # Usually not critical after the fixed dish rectangle is chosen.
    for threshold in DISH_BRIGHT_THRESHOLDS:
        mask = cv2.inRange(gray, threshold, 255)
        mask = cv2.bitwise_and(mask, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            area_frac = area / float(h * w)
            if area_frac < 0.12 or area_frac > 0.55:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            ratio = bw / float(bh)
            if ratio < 0.72 or ratio > 1.28:
                continue
            if bw < w * 0.45 or bw > w * 0.85 or bh < h * 0.35 or bh > h * 0.72:
                continue
            if y > h * 0.18:
                continue
            squareness = 1.0 - min(abs(ratio - 1.0), 0.6)
            score = area * squareness + threshold * 12000.0
            if best_threshold is None or score > best_threshold[0]:
                best_threshold = (score, (x, y, bw, bh))

    if best_threshold is not None:
        x, y, bw, bh = best_threshold[1]
        pad = int(0.01 * max(bw, bh))
        return Dish(
            x=max(0, x - pad),
            y=max(0, y - pad),
            w=min(w - max(0, x - pad), bw + 2 * pad),
            h=min(h - max(0, y - pad), bh + 2 * pad),
            confidence="bright_square",
        )

    upper = np.zeros_like(gray)
    upper[: int(h * 0.78), :] = 255
    edges = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 35, 110)
    edges = cv2.bitwise_and(edges, upper)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[Tuple[float, Tuple[int, int, int, int]]] = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < h * w * 0.05 or area > h * w * 0.70:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if x <= 4 or y <= 4 or x + bw >= w - 4 or y + bh >= h - 4:
            continue
        ratio = bw / float(bh)
        if ratio < 0.70 or ratio > 1.35:
            continue
        if y > h * 0.55:
            continue
        leaf_inside = cv2.countNonZero(leaf_mask[y : y + bh, x : x + bw])
        score = area + 8.0 * leaf_inside - 0.2 * y
        if best is None or score > best[0]:
            best = (score, (x, y, bw, bh))

    if best is not None:
        x, y, bw, bh = best[1]
        pad = int(0.015 * max(bw, bh))
        return Dish(
            x=max(0, x - pad),
            y=max(0, y - pad),
            w=min(w - max(0, x - pad), bw + 2 * pad),
            h=min(h - max(0, y - pad), bh + 2 * pad),
            confidence="edge",
        )

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = [c for c in contours if cv2.contourArea(c) > h * w * MIN_LEAF_AREA_FRAC]
    if large:
        points = np.vstack(large)
        x, y, bw, bh = cv2.boundingRect(points)
        cx, cy = x + bw / 2, y + bh / 2
        side = int(max(bw, bh) * 1.60)
        x0 = max(0, int(cx - side / 2))
        y0 = max(0, int(cy - side / 2))
        return Dish(
            x=x0,
            y=y0,
            w=min(w - x0, side),
            h=min(h - y0, side),
            confidence="leaf_cluster",
        )

    return Dish(x=0, y=0, w=w, h=int(h * 0.78), confidence="fallback")


def parse_dish_arg(value: str) -> Optional[Dish]:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--dish must be four comma-separated integers: x,y,w,h")
    x, y, w, h = [int(float(part)) for part in parts]
    if w <= 0 or h <= 0:
        raise ValueError("--dish width and height must be positive")
    return Dish(x=x, y=y, w=w, h=h, confidence="manual_fixed")


def clamp_dish_to_image(dish: Dish, shape: Tuple[int, int, int]) -> Dish:
    image_h, image_w = shape[:2]
    x = min(max(0, dish.x), image_w - 1)
    y = min(max(0, dish.y), image_h - 1)
    w = min(dish.w, image_w - x)
    h = min(dish.h, image_h - y)
    return Dish(x=x, y=y, w=w, h=h, confidence=dish.confidence)


def estimate_fixed_dish(
    target_files: Sequence[Path],
    calibrations: Dict[Tuple[str, int], Calibration],
    max_images: int = 36,
) -> Optional[Dish]:
    detections: List[Dish] = []
    for path in target_files[:max_images]:
        parsed = parse_image_name(path)
        if parsed is None:
            continue
        img = read_image(path)
        if img is None:
            continue
        calibrated = apply_calibration(img, calibrations.get((parsed.replicate, parsed.dpi)))
        initial_leaf_mask = detect_leaf_mask(calibrated)
        dish = detect_dish(calibrated, initial_leaf_mask)
        if dish.confidence in {"bright_square", "edge"}:
            detections.append(dish)

    if not detections:
        return None

    xs = [dish.x for dish in detections]
    ys = [dish.y for dish in detections]
    ws = [dish.w for dish in detections]
    hs = [dish.h for dish in detections]
    return Dish(
        x=int(round(statistics.median(xs))),
        y=int(round(statistics.median(ys))),
        w=int(round(statistics.median(ws))),
        h=int(round(statistics.median(hs))),
        confidence=f"fixed_median_{len(detections)}",
    )


def extract_leaf_objects_by_grid(
    img_bgr: np.ndarray, leaf_mask: np.ndarray, dish: Dish
) -> List[Tuple[LeafObject, int, int]]:
    objects: List[Tuple[LeafObject, int, int]] = []
    min_area = int(dish.w * dish.h * MIN_LEAF_AREA_FRAC * 0.40)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    exg = 2 * g - r - b

    inner_x0 = dish.x + int(dish.w * 0.015)
    inner_y0 = dish.y + int(dish.h * 0.020)
    inner_x1 = dish.x + dish.w - int(dish.w * 0.010)
    inner_y1 = dish.y + dish.h - int(dish.h * 0.020)
    cell_w = (inner_x1 - inner_x0) / 3.0
    cell_h = (inner_y1 - inner_y0) / 3.0

    for row in range(1, 4):
        for col in range(1, 4):
            x0 = int(round(inner_x0 + (col - 1) * cell_w))
            x1 = int(round(inner_x0 + col * cell_w))
            y0 = int(round(inner_y0 + (row - 1) * cell_h))
            y1 = int(round(inner_y0 + row * cell_h))

            cell_mask = np.zeros_like(leaf_mask)
            cell_mask[y0:y1, x0:x1] = leaf_mask[y0:y1, x0:x1]
            cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates: List[Tuple[float, int, np.ndarray]] = []
            for contour in contours:
                area = int(cv2.contourArea(contour))
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 25 or h < 25:
                    continue
                contour_mask = np.zeros_like(leaf_mask)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                contour_pixels = contour_mask > 0
                greenish = (
                    (hsv[:, :, 0] >= 28)
                    & (hsv[:, :, 0] <= 100)
                    & (hsv[:, :, 1] >= 25)
                    & (exg > -5)
                    & contour_pixels
                )
                green_frac = float(np.count_nonzero(greenish)) / max(1.0, float(area))
                extent = area / float(w * h)
                aspect = w / float(h)
                aspect_penalty = max(0.25, 1.0 - min(abs(math.log(max(aspect, 0.01))), 1.2))
                edge_penalty = 0.65 if x <= x0 + 3 or y <= y0 + 3 or x + w >= x1 - 3 else 1.0
                score = area * (0.35 + green_frac) * max(0.2, extent) * aspect_penalty * edge_penalty
                candidates.append((score, area, contour))

            if not candidates:
                continue

            candidates.sort(key=lambda item: item[0], reverse=True)
            area = candidates[0][1]
            contour = candidates[0][2]
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            x, y, w, h = cv2.boundingRect(contour)
            single_mask = np.zeros_like(leaf_mask)
            cv2.drawContours(single_mask, [contour], -1, 255, -1)
            objects.append(
                (
                    LeafObject(
                        mask=single_mask,
                        contour=contour,
                        area_px=area,
                        centroid_x=float(cx),
                        centroid_y=float(cy),
                        bbox_x=x,
                        bbox_y=y,
                        bbox_w=w,
                        bbox_h=h,
                    ),
                    row,
                    col,
                )
            )

    return objects


def extract_leaf_objects(leaf_mask: np.ndarray, dish: Dish) -> List[LeafObject]:
    dish_mask = np.zeros_like(leaf_mask)
    dish_mask[dish.y : dish.y + dish.h, dish.x : dish.x + dish.w] = 255
    masked = cv2.bitwise_and(leaf_mask, dish_mask)

    min_area = int(dish.w * dish.h * MIN_LEAF_AREA_FRAC)
    max_area = int(dish.w * dish.h * MAX_LEAF_AREA_FRAC)
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects: List[LeafObject] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 30 or h < 30:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        if not (dish.x <= cx <= dish.x + dish.w and dish.y <= cy <= dish.y + dish.h):
            continue
        single_mask = np.zeros_like(leaf_mask)
        cv2.drawContours(single_mask, [contour], -1, 255, -1)
        objects.append(
            LeafObject(
                mask=single_mask,
                contour=contour,
                area_px=area,
                centroid_x=float(cx),
                centroid_y=float(cy),
                bbox_x=x,
                bbox_y=y,
                bbox_w=w,
                bbox_h=h,
            )
        )

    objects.sort(key=lambda obj: obj.area_px, reverse=True)
    objects = objects[:EXPECTED_LEAVES]
    return objects


def assign_grid_positions(objects: Sequence[LeafObject], dish: Dish) -> Dict[int, Tuple[int, int]]:
    if not objects:
        return {}

    centers = np.array(
        [[obj.centroid_x, obj.centroid_y] for obj in objects],
        dtype=np.float32,
    )
    x_min = max(dish.x, float(np.min(centers[:, 0])))
    x_max = min(dish.x + dish.w, float(np.max(centers[:, 0])))
    y_min = max(dish.y, float(np.min(centers[:, 1])))
    y_max = min(dish.y + dish.h, float(np.max(centers[:, 1])))

    if x_max <= x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1

    assignments: Dict[int, Tuple[int, int]] = {}
    taken: set[Tuple[int, int]] = set()
    for idx, obj in sorted(enumerate(objects), key=lambda item: item[1].area_px, reverse=True):
        col_float = (obj.centroid_x - x_min) / (x_max - x_min) * 2.0
        row_float = (obj.centroid_y - y_min) / (y_max - y_min) * 2.0
        col = int(round(col_float)) + 1
        row = int(round(row_float)) + 1
        col = min(3, max(1, col))
        row = min(3, max(1, row))

        if (row, col) in taken:
            options = [
                (r, c)
                for r in range(1, 4)
                for c in range(1, 4)
                if (r, c) not in taken
            ]
            row, col = min(
                options,
                key=lambda rc: (
                    (row_float - (rc[0] - 1)) ** 2 + (col_float - (rc[1] - 1)) ** 2
                ),
            )
        assignments[idx] = (row, col)
        taken.add((row, col))

    return assignments


def classify_leaf_pixels(img_bgr: np.ndarray, mask: np.ndarray) -> Dict[str, int]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(img_bgr.astype(np.int16))

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    a_lab = lab[:, :, 1]
    b_lab = lab[:, :, 2]
    exg = 2 * g - r - b
    inside = mask > 0

    # THRESHOLD PLACEHOLDER: disease class segmentation.
    # Tune HEALTHY_*, CHLOROSIS_*, NECROSIS_*, and DETERIORATION_* constants
    # near the top of this script.
    healthy = (
        inside
        & (h >= HEALTHY_H_MIN)
        & (h <= HEALTHY_H_MAX)
        & (s >= HEALTHY_S_MIN)
        & (v >= HEALTHY_V_MIN)
        & (exg >= HEALTHY_EXG_MIN)
    )
    chlorosis = (
        inside
        & (h >= CHLOROSIS_H_MIN)
        & (h < CHLOROSIS_H_MAX)
        & (s >= CHLOROSIS_S_MIN)
        & (v >= CHLOROSIS_V_MIN)
    )
    necrosis = inside & (
        (
            ((h <= NECROSIS_H_LOW_MAX) | (h >= NECROSIS_H_HIGH_MIN))
            & (s >= NECROSIS_S_MIN)
            & (v <= NECROSIS_V_MAX)
            & (a_lab >= NECROSIS_A_MIN)
        )
        | ((v <= NECROSIS_DARK_V_MAX) & (s >= NECROSIS_DARK_S_MIN))
    )

    classified = healthy | chlorosis | necrosis
    deterioration = (
        inside
        & (~classified)
        & (
            (s < DETERIORATION_S_MAX)
            | (v < DETERIORATION_V_MAX)
            | (b_lab < DETERIORATION_B_MAX)
        )
    )

    # Resolve overlaps with a disease-first order.
    healthy = healthy & (~chlorosis) & (~necrosis)
    chlorosis = chlorosis & (~necrosis)

    total = int(cv2.countNonZero(mask))
    return {
        "total_leaf_px": total,
        "healthy_px": int(np.count_nonzero(healthy)),
        "chlorosis_px": int(np.count_nonzero(chlorosis)),
        "necrosis_px": int(np.count_nonzero(necrosis)),
        "deterioration_px": int(np.count_nonzero(deterioration)),
    }


def contour_metrics(obj: LeafObject) -> Dict[str, float]:
    contour = obj.contour
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    x, y, w, h = cv2.boundingRect(contour)

    major_axis = 0.0
    minor_axis = 0.0
    aspect_ratio = 0.0
    if len(contour) >= 5:
        (_, _), axes, _ = cv2.fitEllipse(contour)
        minor_axis = float(min(axes))
        major_axis = float(max(axes))
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0

    circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0
    extent = area / float(w * h) if w * h > 0 else 0.0

    return {
        "morph_area_px": area,
        "morph_perimeter_px": perimeter,
        "major_axis_px": major_axis,
        "minor_axis_px": minor_axis,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
    }


def pct(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def overlay_visual_check(
    img_bgr: np.ndarray,
    dish: Dish,
    objects: Sequence[LeafObject],
    assignments: Dict[int, Tuple[int, int]],
    out_path: Path,
) -> None:
    overlay = img_bgr.copy()
    cv2.rectangle(
        overlay,
        (dish.x, dish.y),
        (dish.x + dish.w, dish.y + dish.h),
        (255, 0, 255),
        5,
    )

    for idx, obj in enumerate(objects):
        row, col = assignments.get(idx, (0, 0))
        cv2.drawContours(overlay, [obj.contour], -1, (0, 255, 0), 4)
        cv2.rectangle(
            overlay,
            (obj.bbox_x, obj.bbox_y),
            (obj.bbox_x + obj.bbox_w, obj.bbox_y + obj.bbox_h),
            (0, 180, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"{row}:{col}",
            (int(obj.centroid_x) - 35, int(obj.centroid_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            4,
        )

    max_w = 1500
    if overlay.shape[1] > max_w:
        scale = max_w / overlay.shape[1]
        overlay = cv2.resize(
            overlay,
            (int(overlay.shape[1] * scale), int(overlay.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(out_path), overlay)


def find_images(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
        files.extend(input_dir.glob(f"R*/{ext}"))
    return sorted(set(files))


def analyze_image(
    path: Path,
    parsed: ParsedName,
    calibration: Optional[Calibration],
    inventory: Dict[Tuple[str, str, int, int], str],
    visual_dir: Path,
    save_visual: bool,
    fixed_dish: Optional[Dish] = None,
) -> List[Dict[str, object]]:
    img = read_image(path)
    if img is None:
        return []

    calibrated = apply_calibration(img, calibration)
    if fixed_dish is not None:
        dish = clamp_dish_to_image(fixed_dish, calibrated.shape)
        leaf_mask = detect_leaf_mask(calibrated, dish)
    else:
        initial_leaf_mask = detect_leaf_mask(calibrated)
        dish = detect_dish(calibrated, initial_leaf_mask)
        leaf_mask = detect_leaf_mask(calibrated, dish)
    grid_objects = extract_leaf_objects_by_grid(calibrated, leaf_mask, dish)

    rows: List[Dict[str, object]] = []
    plate_full = f"{parsed.replicate}_{parsed.plate_code}"
    if parsed.plate_code.upper().startswith(parsed.replicate + "_"):
        plate_full = parsed.plate_code

    for idx, (obj, grid_row, grid_col) in enumerate(grid_objects):
        accession = inventory.get((parsed.replicate, plate_full, grid_row, grid_col), "")
        counts = classify_leaf_pixels(calibrated, obj.mask)
        metrics = contour_metrics(obj)
        total = counts["total_leaf_px"]
        symptomatic_px = counts["chlorosis_px"] + counts["necrosis_px"]
        disease_px = symptomatic_px
        leaf_id = f"{parsed.replicate}_{plate_full}_{grid_row}:{grid_col}"

        row: Dict[str, object] = {
            "file": path.name,
            "relative_path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            "replicate": parsed.replicate,
            "plate": plate_full,
            "plate_code": parsed.plate_code,
            "dpi": parsed.dpi,
            "leaf_id": leaf_id,
            "grid_row": grid_row,
            "grid_col": grid_col,
            "position": f"{grid_row}:{grid_col}",
            "accession": accession,
            "dish_x": dish.x,
            "dish_y": dish.y,
            "dish_w": dish.w,
            "dish_h": dish.h,
            "dish_detection": dish.confidence,
            "leaf_rank_by_area": idx + 1,
            "leaf_centroid_x": round(obj.centroid_x, 2),
            "leaf_centroid_y": round(obj.centroid_y, 2),
            "leaf_rel_x": round((obj.centroid_x - dish.x) / dish.w, 5) if dish.w else 0,
            "leaf_rel_y": round((obj.centroid_y - dish.y) / dish.h, 5) if dish.h else 0,
            "bbox_x": obj.bbox_x,
            "bbox_y": obj.bbox_y,
            "bbox_w": obj.bbox_w,
            "bbox_h": obj.bbox_h,
            "calibration_method": calibration.method if calibration else "missing",
            "calibration_gain_b": round(float(calibration.gains_bgr[0]), 5) if calibration else 1.0,
            "calibration_gain_g": round(float(calibration.gains_bgr[1]), 5) if calibration else 1.0,
            "calibration_gain_r": round(float(calibration.gains_bgr[2]), 5) if calibration else 1.0,
            **counts,
            "healthy_pct": round(pct(counts["healthy_px"], total), 5),
            "chlorosis_pct": round(pct(counts["chlorosis_px"], total), 5),
            "necrosis_pct": round(pct(counts["necrosis_px"], total), 5),
            "deterioration_pct": round(pct(counts["deterioration_px"], total), 5),
            "symptomatic_px": symptomatic_px,
            "symptomatic_pct": round(pct(symptomatic_px, total), 5),
            "disease_px": disease_px,
            "disease_pct": round(pct(disease_px, total), 5),
            **{k: round(v, 5) for k, v in metrics.items()},
            "leaf_count_detected_in_image": len(grid_objects),
        }
        rows.append(row)

    if save_visual:
        ensure_dir(visual_dir)
        objects = [obj for obj, _, _ in grid_objects]
        assignments = {idx: (row, col) for idx, (_, row, col) in enumerate(grid_objects)}
        overlay_visual_check(
            calibrated,
            dish,
            objects,
            assignments,
            visual_dir / f"check_{path.stem}.jpg",
        )

    if len(grid_objects) != EXPECTED_LEAVES:
        print(
            f"[WARN] {path.name}: detected {len(grid_objects)} leaves "
            f"(expected {EXPECTED_LEAVES}). Check visual overlay."
        )

    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    try:
        handle = path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        print(f"[WARN] Could not overwrite {path.name}; writing {fallback.name} instead.")
        handle = fallback.open("w", newline="", encoding="utf-8")

    with handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def add_progression(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["leaf_id"]), []).append(dict(row))

    out: List[Dict[str, object]] = []
    for leaf_id, leaf_rows in grouped.items():
        leaf_rows.sort(key=lambda r: int(r["dpi"]))
        baseline = leaf_rows[0]
        for row in leaf_rows:
            for key in (
                "disease_pct",
                "symptomatic_pct",
                "chlorosis_pct",
                "necrosis_pct",
                "deterioration_pct",
                "total_leaf_px",
            ):
                base_value = float(baseline[key])
                row[f"baseline_{key}"] = round(base_value, 5)
                row[f"delta_{key}"] = round(float(row[key]) - base_value, 5)

            row["disease_adjusted_pct"] = max(0.0, float(row["delta_disease_pct"]))
            row["symptomatic_adjusted_pct"] = max(0.0, float(row["delta_symptomatic_pct"]))
            row["chlorosis_adjusted_pct"] = max(0.0, float(row["delta_chlorosis_pct"]))
            row["necrosis_adjusted_pct"] = max(0.0, float(row["delta_necrosis_pct"]))
            row["deterioration_adjusted_pct"] = max(0.0, float(row["delta_deterioration_pct"]))
            out.append(row)

    out.sort(key=lambda r: (str(r["replicate"]), str(r["plate"]), str(r["position"]), int(r["dpi"])))
    return out


def trapezoid_audpc(points: Sequence[Tuple[int, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    sorted_points = sorted(points)
    for (x0, y0), (x1, y1) in zip(sorted_points[:-1], sorted_points[1:]):
        total += (x1 - x0) * (y0 + y1) / 2.0
    return total


def build_audpc(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["leaf_id"]), []).append(row)

    out: List[Dict[str, object]] = []
    for leaf_id, leaf_rows in grouped.items():
        leaf_rows = sorted(leaf_rows, key=lambda r: int(r["dpi"]))
        first = leaf_rows[0]
        dpis = [int(r["dpi"]) for r in leaf_rows]
        audpc_row = {
            "leaf_id": leaf_id,
            "replicate": first["replicate"],
            "plate": first["plate"],
            "position": first["position"],
            "grid_row": first["grid_row"],
            "grid_col": first["grid_col"],
            "accession": first["accession"],
            "n_timepoints": len(leaf_rows),
            "first_dpi": min(dpis),
            "last_dpi": max(dpis),
            "mean_leaf_area_px": round(
                statistics.mean(float(r["total_leaf_px"]) for r in leaf_rows), 5
            ),
        }
        for key in (
            "disease_pct",
            "disease_adjusted_pct",
            "chlorosis_pct",
            "necrosis_pct",
            "deterioration_pct",
        ):
            points = [(int(r["dpi"]), float(r[key])) for r in leaf_rows]
            audpc_row[f"audpc_{key}"] = round(trapezoid_audpc(points), 5)
        out.append(audpc_row)

    out.sort(key=lambda r: (str(r["replicate"]), str(r["plate"]), str(r["position"])))
    return out


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    inventory_path = Path(args.inventory).resolve() if args.inventory else input_dir / INVENTORY_NAME
    visual_dir = output_dir / "visual_checks"

    ensure_dir(output_dir)
    image_files = find_images(input_dir)
    if not image_files:
        raise SystemExit(f"No images found in {input_dir}")

    inventory = load_inventory(inventory_path)
    print(f"Loaded {len(inventory)} inventory plate-position records from {inventory_path.name}")

    calibrations = build_calibrations(image_files)
    print(f"Built {len(calibrations)} replicate/day color calibrations")

    target_files = [
        path
        for path in image_files
        if (parsed := parse_image_name(path)) is not None and not parsed.is_datacolor
    ]
    if args.match:
        needle = args.match.lower()
        target_files = [path for path in target_files if needle in path.stem.lower()]
    if args.limit and args.limit > 0:
        target_files = target_files[: args.limit]

    fixed_dish = None
    if not args.per_image_dish:
        fixed_dish = parse_dish_arg(args.dish)
        if fixed_dish is None:
            fixed_dish = estimate_fixed_dish(target_files, calibrations)
        if fixed_dish is None:
            print("[WARN] Could not estimate a fixed Petri dish rectangle; using per-image detection.")
        else:
            print(
                "Using fixed Petri dish rectangle: "
                f"x={fixed_dish.x}, y={fixed_dish.y}, w={fixed_dish.w}, h={fixed_dish.h} "
                f"({fixed_dish.confidence})"
            )

    all_rows: List[Dict[str, object]] = []
    saved_visuals = 0
    for i, path in enumerate(target_files, start=1):
        parsed = parse_image_name(path)
        if parsed is None:
            continue
        calibration = calibrations.get((parsed.replicate, parsed.dpi))
        save_visual = (
            VISUAL_CHECKS
            and not args.no_visual_checks
            and (args.max_visual_checks == 0 or saved_visuals < args.max_visual_checks)
        )
        rows = analyze_image(
            path,
            parsed,
            calibration,
            inventory,
            visual_dir,
            save_visual,
            fixed_dish=fixed_dish,
        )
        all_rows.extend(rows)
        if save_visual:
            saved_visuals += 1
        print(f"[{i}/{len(target_files)}] {path.name}: {len(rows)} leaf records")

    measurements_path = output_dir / "inoc_iv_leaf_measurements.csv"
    progression_path = output_dir / "inoc_iv_leaf_progression.csv"
    audpc_path = output_dir / "inoc_iv_leaf_audpc.csv"

    write_csv(measurements_path, all_rows)
    progression_rows = add_progression(all_rows)
    write_csv(progression_path, progression_rows)
    write_csv(audpc_path, build_audpc(progression_rows))

    print("\nDone.")
    print(f"Measurements: {measurements_path}")
    print(f"Progression:  {progression_path}")
    print(f"AUDPC:        {audpc_path}")
    if VISUAL_CHECKS and not args.no_visual_checks:
        print(f"Visual checks: {visual_dir}")


if __name__ == "__main__":
    main()
