"""Segment and track every plate of one scanner, straight from the raw scans.

Per frame, the scan is read once (memory-mapped). Every plate is cropped at its aligned
position, segmented by the chosen detector, and its colonies are tracked. The default
footprint tracker (``scanlag.track.tracker``) keeps each colony's own pixels and splits merged
blobs between their colonies; ``tracker="seed"`` is MATLAB's seed-pixel tracker
(``scanlag.track.seed_tracker``), where merged colonies all report the blob. Outputs per plate,
under ``<results>/<scanner>/plate_<k>/``: plate.json, mask.png, labels.zarr (detected objects),
colonies.zarr (colony IDs, footprint tracker only), tracks.parquet, colonies.parquet (see
``write_plate``). ``retrack_plate`` tracks an already segmented plate again from its labels.zarr.

Detectors:
* ``classical``: MATLAB's threshold (im2L.m), with contrast limits from the plate's last frame
* a trained model directory (U-Net or Cellpose-SAM, ``scanlag.detect.unet.load_detector``)
"""

from __future__ import annotations

import collections
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm import tqdm

from scanlag.detect.classical import DEFAULT_THRESHOLD, binarize, label_column_major, stretch_limits, subtract_background
from scanlag.io.matlab_import import ColonyStatus
from scanlag.io.raw import read_scan
from scanlag.pipeline.flags import colony_flags
from scanlag.pipeline.plates import Aligner, Plate, circle_mask, crop, find_plates, is_empty_slot
from scanlag.pipeline.scans import Scanner
from scanlag.store import ResultsStore
from scanlag.testset.tiles import PREVIOUS_GAP
from scanlag.track.seed_tracker import MIN_AREA, SeedTracker, border_status
from scanlag.track.tracker import FootprintTracker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "unet": PROJECT_ROOT / "models" / "unet_scanlag_v1",
    "cellpose": PROJECT_ROOT / "models" / "cpsam_scanlag_v1",
}


@dataclass
class RunConfig:
    detector: str = "unet"  # "classical", "unet", "cellpose" or a model directory
    n_plates: int = 6
    mask_fraction: float = 0.8
    threshold: float = DEFAULT_THRESHOLD  # classical only
    min_upper_limit: float = 0.06  # classical only: floor on the contrast stretch's upper limit
    min_area: int = MIN_AREA  # px; smaller unmatched blobs don't start a colony
    tracker: str = "footprint"  # or "seed" (MATLAB: merged colonies report the merged blob)


class ClassicalSegmenter:
    """MATLAB's threshold detector.

    Images2Colonies.m stretches contrast to the cleaned last frame's 1-99 % range inside the mask,
    so the effective threshold is TH x upper limit. On an empty or sparse plate the upper limit
    is only a few gray levels and noise floods the plate (tens of thousands of blobs), and a few
    pixels' difference in the mask moves it by one gray level. ``min_upper_limit`` puts a floor
    under it (0 = exact MATLAB behaviour).
    """

    name = "classical"

    def __init__(self, threshold: float, min_upper_limit: float = 0.0, log=print):
        self.threshold = threshold
        self.min_upper_limit = min_upper_limit
        self.limits: dict[int, np.ndarray] = {}
        self.log = log

    def set_last_frame(self, plate: int, last: np.ndarray, background: np.ndarray, mask: np.ndarray) -> None:
        lo, hi = stretch_limits(subtract_background(last, background)[mask])
        if hi < self.min_upper_limit:
            self.log(f"  plate {plate}: contrast upper limit {hi:.3f} raised to {self.min_upper_limit:.3f}")
            hi = self.min_upper_limit
        self.limits[plate] = np.array([lo, hi])

    def segment(self, plate: int, image, background, previous, mask) -> np.ndarray:
        labels, _ = label_column_major(binarize(image, background, self.limits[plate], self.threshold, mask))
        return labels


class ModelSegmenter:
    def __init__(self, model_dir: Path):
        from scanlag.detect.unet import load_detector

        self.det = load_detector(model_dir)
        self.name = Path(model_dir).name

    def segment(self, plate: int, image, background, previous, mask) -> np.ndarray:
        labels = self.det.segment([self.det.prepare(image, background, previous)])[0].astype(np.int32)
        ids = np.unique(labels)[1:]
        if len(ids):  # keep objects whose centroid is in the working area, renumbered 1..n
            cy, cx = np.array(ndi.center_of_mass(np.ones_like(labels), labels, ids)).round().astype(int).T
            keep = ids[mask[np.clip(cy, 0, mask.shape[0] - 1), np.clip(cx, 0, mask.shape[1] - 1)]]
            lut = np.zeros(labels.max() + 1, np.int32)
            lut[keep] = np.arange(1, len(keep) + 1)
            labels = lut[labels]
        return labels


def make_segmenter(cfg: RunConfig, log=print):
    if cfg.detector == "classical":
        return ClassicalSegmenter(cfg.threshold, cfg.min_upper_limit, log)
    return ModelSegmenter(MODELS.get(cfg.detector, Path(cfg.detector)))


def make_tracker(cfg: RunConfig):
    if cfg.tracker == "footprint":
        return FootprintTracker(cfg.min_area)
    if cfg.tracker == "seed":
        return SeedTracker(cfg.min_area)
    raise ValueError(f"unknown tracker {cfg.tracker!r} (footprint or seed)")


def process_scanner(scanner: Scanner, store: ResultsStore, cfg: RunConfig, log=print) -> list[Path]:
    t0 = time.time()
    n = len(scanner.scans)
    background_scan = read_scan(scanner.scans[0])
    plates = find_plates(background_scan, cfg.n_plates)
    empty = [p.number for p in plates if is_empty_slot(background_scan, p)]
    if empty:
        log(f"{scanner.name}: plates {empty} are empty slots (no dish), skipped")
    plates = [p for p in plates if p.number not in empty]
    aligners = {p.number: Aligner(background_scan, p, mask_fraction=cfg.mask_fraction) for p in plates}
    size = {p.number: p.crop[2] for p in plates}
    masks = {p.number: circle_mask(size[p.number], p, cfg.mask_fraction) for p in plates}
    backgrounds = {p.number: crop(background_scan, *p.crop) for p in plates}
    seg = make_segmenter(cfg, log)
    log(f"{scanner.name}: {n} scans, plates at " + ", ".join(f"{p.number}:({p.cx:.0f},{p.cy:.0f})" for p in plates))

    def plate_crop(image, k, plate):
        dx, dy = (0, 0) if k == 0 else aligners[plate.number].shift(image)
        x0, y0, s = plate.crop
        return crop(image, x0 + dx, y0 + dy, s), (dx, dy)

    if isinstance(seg, ClassicalSegmenter):
        last = read_scan(scanner.scans[-1])
        for p in plates:
            seg.set_last_frame(p.number, plate_crop(last, n - 1, p)[0], backgrounds[p.number], masks[p.number])

    out = {p.number: (scanner.name, f"plate_{p.number}") for p in plates}
    labels_out = {p.number: store.create_labels((n, size[p.number], size[p.number]), *out[p.number], "labels.zarr") for p in plates}
    trackers = {p.number: make_tracker(cfg) for p in plates}
    colonies_out = {}
    if cfg.tracker == "footprint":
        colonies_out = {p.number: store.create_labels((n, size[p.number], size[p.number]), *out[p.number], "colonies.zarr")
                        for p in plates}
    shifts = {p.number: [] for p in plates}
    history = {p.number: collections.deque(maxlen=PREVIOUS_GAP + 1) for p in plates}

    io = ThreadPoolExecutor(1)
    pending = io.submit(lambda: np.asarray(background_scan))
    bar = tqdm(total=n, desc=f"{scanner.name} ({len(plates)} plates)", unit="scan", dynamic_ncols=True)
    for k in range(n):
        image = pending.result()
        if k + 1 < n:  # prefetch the next scan while this one is processed
            pending = io.submit(lambda j=k + 1: np.asarray(read_scan(scanner.scans[j])))
        for p in plates:
            im, shift = plate_crop(image, k, p)
            shifts[p.number].append(shift)
            hist = history[p.number]
            hist.append(im)
            labels = seg.segment(p.number, im, backgrounds[p.number], hist[0], masks[p.number])
            labels_out[p.number][k] = labels.astype(np.uint16)
            own = trackers[p.number].update(labels, int(labels.max()))
            if p.number in colonies_out:
                colonies_out[p.number][k] = own.astype(np.uint16)
        bar.update()
    bar.close()
    log(f"{scanner.name}: segmented in {(time.time() - t0) / 60:.1f} min")

    dirs = []
    for p in plates:
        dirs.append(write_plate(store, out[p.number], scanner, p, cfg, seg, masks[p.number], shifts[p.number],
                                trackers[p.number], labels_out[p.number], colonies_out.get(p.number)))
    return dirs


def write_plate(store, out, scanner: Scanner, plate: Plate, cfg: RunConfig, seg, mask, shifts,
                tracker, labels, own=None) -> Path:
    detector = seg.name
    times = np.asarray(scanner.times_min)
    tracks, colonies = track_tables(tracker, times, mask, labels, own)
    x0, y0, s = plate.crop
    store.write_json({
        "scanner": scanner.name, "scanner_label": scanner.label, "plate": plate.number,
        "raw_dir": scanner.directory, "raw_scans": [p.name for p in scanner.scans],
        "times_min": times, "start_time": scanner.start.isoformat(sep=" "),
        "circle": plate.to_dict(), "crop_xy_size": [x0, y0, s],
        "shift_xy": shifts, "mask_fraction": cfg.mask_fraction,
        "detector": detector, "threshold": cfg.threshold if detector == "classical" else None,
        "contrast_limits": getattr(seg, "limits", {}).get(plate.number),
        "min_area": cfg.min_area, "tracker": cfg.tracker,
    }, *out, "plate.json")
    store.write_mask(mask, *out, "mask.png")
    store.write_table(tracks, *out, "tracks.parquet")
    store.write_table(colonies, *out, "colonies.parquet")
    return store.path(*out)


def track_tables(tracker, times: np.ndarray, mask: np.ndarray, labels, own=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tracks, colonies) tables of a plate from its tracker, with automatic statuses and flags."""
    area, centroid, blob = tracker.tracks.as_arrays()
    n_frames, n_col = area.shape
    # merged: the colony shares its blob with another colony. With the seed tracker its area is
    # then the blob's; the footprint tracker gives it its share of the blob.
    merged = np.zeros_like(area, bool)
    for k in range(n_frames):
        b = blob[k]
        counts = np.bincount(b[b > 0], minlength=1)
        merged[k] = (b > 0) & (counts[b] > 1)

    t_idx, c_idx = np.nonzero(area)
    cx, cy = centroid[t_idx, c_idx, 0] - 1, centroid[t_idx, c_idx, 1] - 1
    tracks = pd.DataFrame({
        "frame": t_idx, "time_min": times[t_idx], "colony_id": c_idx + 1, "area_px": area[t_idx, c_idx],
        "x": cx, "y": cy, "blob": blob[t_idx, c_idx], "merged": merged[t_idx, c_idx],
    })

    seeds = tracker.tracks.seeds
    status = np.full(n_col, int(ColonyStatus.VALID))
    if n_col:
        status[~border_status(mask, seeds)] = ColonyStatus.NEAR_BORDER
        status[area[-1] == 0] = ColonyStatus.ZERO_FINAL_AREA
    appear = np.argmax(area > 0, axis=0) if n_col else np.zeros(0, int)
    ever_merged = merged.any(axis=0)
    colonies = pd.DataFrame({
        "colony_id": np.arange(1, n_col + 1),
        "status": status,
        "status_name": [ColonyStatus(s).name for s in status],
        "auto_status": status,
        "appearance_frame": appear,
        "appearance_min": times[appear] if n_col else [],
        "first_merged_frame": np.where(ever_merged, np.argmax(merged, axis=0), -1) if n_col else [],
        "final_area_px": area[-1] if n_col else [],
        "seed_x": [c for _, c in seeds],
        "seed_y": [r for r, _ in seeds],
    })
    colonies = colonies.join(colony_flags(area, merged, blob, labels, own).set_index("colony_id"), on="colony_id")
    return tracks, colonies


def retrack_plate(store: ResultsStore, plate_dir: Path, tracker: str = "footprint") -> dict:
    """Track an already segmented plate again from its labels.zarr, without reading the scans.

    Rewrites tracks.parquet, colonies.parquet, colonies.zarr (footprint tracker) and the tracker in
    plate.json. Colony IDs change, so review decisions can't carry over: the old colonies.parquet and
    review.json are kept as colonies.<old tracker>.bak.parquet and review.<old tracker>.bak.json, and
    the plate is reviewed afresh. Returns the old and new colony counts.
    """
    import zarr
    from PIL import Image

    plate_dir = Path(plate_dir)
    rel = plate_dir.resolve().relative_to(store.root).parts
    info = plate_info(plate_dir)
    cfg = RunConfig(tracker=tracker, min_area=info.get("min_area", MIN_AREA))
    labels = zarr.open_array(str(plate_dir / "labels.zarr"), mode="r")
    mask = np.array(Image.open(plate_dir / "mask.png")).astype(bool)
    times = np.asarray(info["times_min"], float)

    old = info.get("tracker", "seed")
    n_old = len(pd.read_parquet(plate_dir / "colonies.parquet"))
    for name, suffix in [("colonies", ".parquet"), ("review", ".json")]:
        if (plate_dir / f"{name}{suffix}").exists():
            (plate_dir / f"{name}{suffix}").replace(store.path(*rel, f"{name}.{old}.bak{suffix}"))
    if (plate_dir / "colonies.zarr").exists():
        shutil.rmtree(store.path(*rel, "colonies.zarr"))

    tr = make_tracker(cfg)
    own_out = store.create_labels(labels.shape, *rel, "colonies.zarr") if tracker == "footprint" else None
    for k in range(labels.shape[0]):
        lab = np.asarray(labels[k]).astype(np.int32)
        own = tr.update(lab, int(lab.max()))
        if own_out is not None:
            own_out[k] = own.astype(np.uint16)
    tracks, colonies = track_tables(tr, times, mask, labels, own_out)
    store.write_json(info | {"tracker": tracker, "min_area": cfg.min_area}, *rel, "plate.json")
    store.write_table(tracks, *rel, "tracks.parquet")
    store.write_table(colonies, *rel, "colonies.parquet")
    return {"old_tracker": old, "old_colonies": n_old, "colonies": len(colonies),
            "flagged": int((colonies.n_flags > 0).sum())}


def refresh_flags(plate_dir: Path) -> int:
    """Recompute the review flags of a processed plate (e.g. after the flag rules changed).

    Statuses and review decisions are kept. Returns the number of flagged colonies.
    """
    import zarr

    colonies = pd.read_parquet(plate_dir / "colonies.parquet")
    tracks = pd.read_parquet(plate_dir / "tracks.parquet")
    labels = zarr.open_array(str(plate_dir / "labels.zarr"), mode="r")
    own = zarr.open_array(str(plate_dir / "colonies.zarr"), mode="r") if merged_area_is_own(plate_info(plate_dir)) else None
    shape = (labels.shape[0], len(colonies))
    area, merged, blob = np.zeros(shape, np.int64), np.zeros(shape, bool), np.zeros(shape, np.int64)
    t, c = tracks.frame.to_numpy(), tracks.colony_id.to_numpy() - 1
    area[t, c], merged[t, c], blob[t, c] = tracks.area_px, tracks.merged, tracks.blob
    flags = colony_flags(area, merged, blob, labels, own).set_index("colony_id")
    colonies = colonies.drop(columns=["flags", "n_flags"]).join(flags, on="colony_id")
    colonies.to_parquet(plate_dir / "colonies.parquet", index=False)
    return int((colonies.n_flags > 0).sum())


def load_plate_dirs(results: Path) -> list[Path]:
    return sorted(p.parent for p in Path(results).glob("*/plate_*/colonies.parquet"))


def plate_info(plate_dir: Path) -> dict:
    return json.loads((Path(plate_dir) / "plate.json").read_text())


def merged_area_is_own(info: dict) -> bool:
    """True if a merged colony's area is its own share (footprint tracker), not the merged blob's.
    Plates processed before the footprint tracker existed have no "tracker" key."""
    return info.get("tracker", "seed") == "footprint"
