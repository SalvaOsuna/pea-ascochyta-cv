"""
Manual threshold selector for the Inoc_IV pea leaf image pipeline.

Run from RStudio:

    reticulate::py_run_file("py/manual_threshold_selector_inoc_iv.py")

Or from a terminal:

    python py/manual_threshold_selector_inoc_iv.py --image Inoc_IV/R1/R1_A_3dpi.jpg

Useful keys while the window is open:

    p   print the current threshold constants
    s   save the current visual panel as a PNG
    q   quit

The printed constants can be pasted into the
"VISUAL THRESHOLD PLACEHOLDERS" block in py/analyze_inoc_iv.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from analyze_inoc_iv import (
        Calibration,
        apply_calibration,
        datacolor_white_patch,
        detect_dish,
        parse_image_name,
    )
except ImportError:
    Calibration = None
    apply_calibration = None
    datacolor_white_patch = None
    detect_dish = None
    parse_image_name = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "Inoc_IV" / "R1" / "R1_A_8dpi.jpg"
OUTPUT_DIR = REPO_ROOT / "visual_checks"


def empty(_: int) -> None:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive threshold selector for Inoc_IV images.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Image to inspect.")
    parser.add_argument(
        "--dish",
        default="",
        help="Optional Petri dish rectangle as x,y,w,h pixels. If omitted, auto-detects.",
    )
    parser.add_argument("--scale", type=int, default=35, help="Display scale percent.")
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Inspect the raw image instead of applying the matching Datacolor correction.",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    return img


def parse_dish(value: str) -> Optional[Tuple[int, int, int, int]]:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise SystemExit("--dish must be x,y,w,h")
    x, y, w, h = [int(float(part)) for part in parts]
    return x, y, w, h


def matching_datacolor_path(image_path: Path) -> Optional[Path]:
    if parse_image_name is None:
        return None
    parsed = parse_image_name(image_path)
    if parsed is None or parsed.is_datacolor:
        return None
    candidate = image_path.parent / f"{parsed.replicate}_0Datacolor_{parsed.dpi}dpi{image_path.suffix}"
    return candidate if candidate.exists() else None


def maybe_calibrate(img_bgr: np.ndarray, image_path: Path, no_calibration: bool) -> np.ndarray:
    if no_calibration or datacolor_white_patch is None or apply_calibration is None:
        return img_bgr
    datacolor_path = matching_datacolor_path(image_path)
    if datacolor_path is None:
        print("[WARN] No matching Datacolor image found; using raw image.")
        return img_bgr
    checker = cv2.imread(str(datacolor_path))
    if checker is None:
        print("[WARN] Could not read matching Datacolor image; using raw image.")
        return img_bgr
    gains, method = datacolor_white_patch(checker)
    calibration = Calibration(
        replicate="manual",
        dpi=0,
        gains_bgr=gains,
        white_patch_bgr=(0.0, 0.0, 0.0),
        method=method,
    )
    print(f"Applied Datacolor correction from {datacolor_path.name} ({method}).")
    return apply_calibration(img_bgr, calibration)


def resize_for_display(img: np.ndarray, scale_percent: int) -> np.ndarray:
    if scale_percent == 100:
        return img.copy()
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def create_trackbars() -> None:
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 720, 720)

    cv2.createTrackbar("View 0All 1Leaf 2Healthy 3Chlor 4Nec 5Deter", "Controls", 0, 5, empty)

    # Leaf mask thresholds.
    cv2.createTrackbar("LEAF_BSTAR_MIN", "Controls", 132, 255, empty)
    cv2.createTrackbar("LEAF_S_MIN", "Controls", 22, 255, empty)
    cv2.createTrackbar("LEAF_V_MIN", "Controls", 35, 255, empty)
    cv2.createTrackbar("LEAF_V_MAX", "Controls", 245, 255, empty)
    cv2.createTrackbar("LEAF_OPEN_KERNEL", "Controls", 5, 31, empty)
    cv2.createTrackbar("LEAF_CLOSE_KERNEL", "Controls", 11, 51, empty)

    # Healthy tissue thresholds.
    cv2.createTrackbar("HEALTHY_H_MIN", "Controls", 38, 179, empty)
    cv2.createTrackbar("HEALTHY_H_MAX", "Controls", 96, 179, empty)
    cv2.createTrackbar("HEALTHY_S_MIN", "Controls", 32, 255, empty)
    cv2.createTrackbar("HEALTHY_V_MIN", "Controls", 45, 255, empty)
    cv2.createTrackbar("HEALTHY_EXG_MIN+255", "Controls", 245, 510, empty)

    # Chlorosis thresholds.
    cv2.createTrackbar("CHLOROSIS_H_MIN", "Controls", 18, 179, empty)
    cv2.createTrackbar("CHLOROSIS_H_MAX", "Controls", 38, 179, empty)
    cv2.createTrackbar("CHLOROSIS_S_MIN", "Controls", 25, 255, empty)
    cv2.createTrackbar("CHLOROSIS_V_MIN", "Controls", 65, 255, empty)

    # Necrosis thresholds.
    cv2.createTrackbar("NECROSIS_H_LOW_MAX", "Controls", 24, 179, empty)
    cv2.createTrackbar("NECROSIS_H_HIGH_MIN", "Controls", 165, 179, empty)
    cv2.createTrackbar("NECROSIS_S_MIN", "Controls", 25, 255, empty)
    cv2.createTrackbar("NECROSIS_V_MAX", "Controls", 170, 255, empty)
    cv2.createTrackbar("NECROSIS_A_MIN", "Controls", 124, 255, empty)
    cv2.createTrackbar("NECROSIS_DARK_V_MAX", "Controls", 70, 255, empty)
    cv2.createTrackbar("NECROSIS_DARK_S_MIN", "Controls", 12, 255, empty)

    # Deterioration thresholds.
    cv2.createTrackbar("DETERIORATION_S_MAX", "Controls", 45, 255, empty)
    cv2.createTrackbar("DETERIORATION_V_MAX", "Controls", 75, 255, empty)
    cv2.createTrackbar("DETERIORATION_B_MAX", "Controls", 125, 255, empty)


def get(name: str) -> int:
    return cv2.getTrackbarPos(name, "Controls")


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def current_thresholds() -> dict:
    return {
        "LEAF_BSTAR_MIN": get("LEAF_BSTAR_MIN"),
        "LEAF_S_MIN": get("LEAF_S_MIN"),
        "LEAF_V_MIN": get("LEAF_V_MIN"),
        "LEAF_V_MAX": get("LEAF_V_MAX"),
        "LEAF_OPEN_KERNEL": odd_kernel(get("LEAF_OPEN_KERNEL")),
        "LEAF_CLOSE_KERNEL": odd_kernel(get("LEAF_CLOSE_KERNEL")),
        "HEALTHY_H_MIN": get("HEALTHY_H_MIN"),
        "HEALTHY_H_MAX": get("HEALTHY_H_MAX"),
        "HEALTHY_S_MIN": get("HEALTHY_S_MIN"),
        "HEALTHY_V_MIN": get("HEALTHY_V_MIN"),
        "HEALTHY_EXG_MIN": get("HEALTHY_EXG_MIN+255") - 255,
        "CHLOROSIS_H_MIN": get("CHLOROSIS_H_MIN"),
        "CHLOROSIS_H_MAX": get("CHLOROSIS_H_MAX"),
        "CHLOROSIS_S_MIN": get("CHLOROSIS_S_MIN"),
        "CHLOROSIS_V_MIN": get("CHLOROSIS_V_MIN"),
        "NECROSIS_H_LOW_MAX": get("NECROSIS_H_LOW_MAX"),
        "NECROSIS_H_HIGH_MIN": get("NECROSIS_H_HIGH_MIN"),
        "NECROSIS_S_MIN": get("NECROSIS_S_MIN"),
        "NECROSIS_V_MAX": get("NECROSIS_V_MAX"),
        "NECROSIS_A_MIN": get("NECROSIS_A_MIN"),
        "NECROSIS_DARK_V_MAX": get("NECROSIS_DARK_V_MAX"),
        "NECROSIS_DARK_S_MIN": get("NECROSIS_DARK_S_MIN"),
        "DETERIORATION_S_MAX": get("DETERIORATION_S_MAX"),
        "DETERIORATION_V_MAX": get("DETERIORATION_V_MAX"),
        "DETERIORATION_B_MAX": get("DETERIORATION_B_MAX"),
    }


def print_thresholds() -> None:
    values = current_thresholds()
    print("\n# Paste these into py/analyze_inoc_iv.py")
    for key, value in values.items():
        print(f"{key} = {value}")
    print()


def make_masks(img_bgr: np.ndarray, dish_rect: Tuple[int, int, int, int], values: dict) -> dict:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(img_bgr.astype(np.int16))

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    a_lab = lab[:, :, 1]
    b_lab = lab[:, :, 2]
    exg = 2 * g - r - b

    x, y, w, hh = dish_rect
    dish_mask = np.zeros(h.shape, dtype=np.uint8)
    dish_mask[y : y + hh, x : x + w] = 255
    inside_dish = dish_mask > 0

    leaf = (
        inside_dish
        & (b_lab >= values["LEAF_BSTAR_MIN"])
        & (s >= values["LEAF_S_MIN"])
        & (v >= values["LEAF_V_MIN"])
        & (v <= values["LEAF_V_MAX"])
    ).astype(np.uint8) * 255
    leaf = cv2.medianBlur(leaf, 5)
    open_k = values["LEAF_OPEN_KERNEL"]
    close_k = values["LEAF_CLOSE_KERNEL"]
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    inside_leaf = leaf > 0

    healthy = (
        inside_leaf
        & (h >= values["HEALTHY_H_MIN"])
        & (h <= values["HEALTHY_H_MAX"])
        & (s >= values["HEALTHY_S_MIN"])
        & (v >= values["HEALTHY_V_MIN"])
        & (exg >= values["HEALTHY_EXG_MIN"])
    )
    chlorosis = (
        inside_leaf
        & (h >= values["CHLOROSIS_H_MIN"])
        & (h < values["CHLOROSIS_H_MAX"])
        & (s >= values["CHLOROSIS_S_MIN"])
        & (v >= values["CHLOROSIS_V_MIN"])
    )
    necrosis = inside_leaf & (
        (
            ((h <= values["NECROSIS_H_LOW_MAX"]) | (h >= values["NECROSIS_H_HIGH_MIN"]))
            & (s >= values["NECROSIS_S_MIN"])
            & (v <= values["NECROSIS_V_MAX"])
            & (a_lab >= values["NECROSIS_A_MIN"])
        )
        | ((v <= values["NECROSIS_DARK_V_MAX"]) & (s >= values["NECROSIS_DARK_S_MIN"]))
    )

    healthy = healthy & (~chlorosis) & (~necrosis)
    chlorosis = chlorosis & (~necrosis)
    classified = healthy | chlorosis | necrosis
    deterioration = inside_leaf & (~classified) & (
        (s < values["DETERIORATION_S_MAX"])
        | (v < values["DETERIORATION_V_MAX"])
        | (b_lab < values["DETERIORATION_B_MAX"])
    )

    return {
        "leaf": leaf,
        "healthy": healthy.astype(np.uint8) * 255,
        "chlorosis": chlorosis.astype(np.uint8) * 255,
        "necrosis": necrosis.astype(np.uint8) * 255,
        "deterioration": deterioration.astype(np.uint8) * 255,
    }


def overlay_masks(img_bgr: np.ndarray, masks: dict, dish_rect: Tuple[int, int, int, int], view: int) -> np.ndarray:
    if view == 1:
        panel = cv2.cvtColor(masks["leaf"], cv2.COLOR_GRAY2BGR)
    elif view in {2, 3, 4, 5}:
        names = {2: "healthy", 3: "chlorosis", 4: "necrosis", 5: "deterioration"}
        panel = cv2.bitwise_and(img_bgr, img_bgr, mask=masks[names[view]])
    else:
        color = np.zeros_like(img_bgr)
        color[masks["healthy"] > 0] = (40, 180, 40)
        color[masks["chlorosis"] > 0] = (0, 220, 255)
        color[masks["necrosis"] > 0] = (0, 0, 255)
        color[masks["deterioration"] > 0] = (255, 120, 0)
        panel = cv2.addWeighted(img_bgr, 0.68, color, 0.55, 0)

    x, y, w, h = dish_rect
    cv2.rectangle(panel, (x, y), (x + w, y + h), (255, 0, 255), 4)

    contours, _ = cv2.findContours(masks["leaf"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (0, 255, 0), 2)
    return panel


def auto_dish(img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    if detect_dish is None:
        return 0, 0, img_bgr.shape[1], img_bgr.shape[0]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    initial_leaf = ((lab[:, :, 2] >= 132) & (hsv[:, :, 1] >= 22)).astype(np.uint8) * 255
    dish = detect_dish(img_bgr, initial_leaf)
    return dish.x, dish.y, dish.w, dish.h


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).resolve()
    img = read_image(image_path)
    img = maybe_calibrate(img, image_path, args.no_calibration)

    dish_rect = parse_dish(args.dish)
    if dish_rect is None:
        dish_rect = auto_dish(img)
    print(f"Using dish rectangle x,y,w,h = {dish_rect}")

    display_img = resize_for_display(img, args.scale)
    scale = args.scale / 100.0
    display_dish = tuple(int(round(value * scale)) for value in dish_rect)

    create_trackbars()
    cv2.namedWindow("Threshold selector", cv2.WINDOW_NORMAL)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Press p to print thresholds, s to save the current panel, q to quit.")

    last_panel = display_img.copy()
    while True:
        values = current_thresholds()
        view = get("View 0All 1Leaf 2Healthy 3Chlor 4Nec 5Deter")
        masks = make_masks(display_img, display_dish, values)
        last_panel = overlay_masks(display_img, masks, display_dish, view)

        cv2.imshow("Threshold selector", last_panel)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            print_thresholds()
        if key == ord("s"):
            out_path = OUTPUT_DIR / f"manual_threshold_{image_path.stem}.png"
            cv2.imwrite(str(out_path), last_panel)
            print(f"Saved {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
