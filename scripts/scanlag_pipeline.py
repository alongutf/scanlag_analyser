"""ScanLag analysis pipeline: raw scans -> colonies -> review GUI -> result tables.

    python scripts/scanlag_pipeline.py <scan folder> [--detector unet|cellpose|classical|<model dir>]

1. Finds every scanner's time-lapse in <scan folder> (also in subfolders; MATLAB ROI folders are
   skipped). Scans must be named <prefix>_<scanner>_<YYYYMMDD>_<HHMM>.tif[f]. The first scan of each
   scanner is the background.
2. Finds the plates (default 6 per scanner, numbered as in MATLAB: left column top to bottom,
   then the right column), aligns every frame, segments it and tracks the colonies.
3. Opens the review GUI (PlateAnalyzer-style) to include/exclude colonies, with suspicious tracks
   and shapes flagged.
4. "Finish & export" in the GUI writes the tables to results_<folder name>/<scanner>/plate_<k>/.

The raw folder is only read. Re-running skips the segmentation of scanners that are already
done (use --redo to repeat it), so --review-only or a plain re-run reopens the GUI with the
decisions made so far.

Options:
  --detector       unet (default; U-Net v1, ~0.15 s per plate frame), cellpose (Cellpose-SAM v1,
                   ~5.5 s per plate frame), classical (MATLAB threshold) or a trained model directory
  --out            results directory (default: <project>/results_<scan folder name>)
  --plates         plates per scanner (default 6)
  --mask-fraction  working area radius / plate radius (default 0.8, as MATLAB)
  --threshold      classical threshold TH (default 0.2)
  --tracker        footprint (default: colonies keep their own area after merging) or seed (MATLAB:
                   merged colonies all report the merged blob's area)
  --redo           segment again even if results exist
  --review-only    skip segmentation, open the GUI on existing results
  --reflag         recompute the review flags of existing results (keeps review decisions)
  --retrack        track existing results again with --tracker, from the stored segmentation (no
                   scans read). Colony numbers change, so the review starts afresh; the old
                   colonies.parquet and review.json are kept as *.bak files
  --no-review      segment and export without opening the GUI (all colonies as detected)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from scanlag.pipeline.export import export
from scanlag.pipeline.run import RunConfig, load_plate_dirs, process_scanner, refresh_flags, retrack_plate
from scanlag.pipeline.scans import find_scanners
from scanlag.cli import split_prompt_args
from scanlag.store import ResultsStore, default_results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scans", type=Path, help="folder with the raw scans")
    parser.add_argument("--detector", default="unet")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plates", type=int, default=6)
    parser.add_argument("--mask-fraction", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--tracker", choices=["footprint", "seed"], default="footprint")
    parser.add_argument("--workers", type=int, default=6, help="plates processed in parallel (1 = serial)")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--reflag", action="store_true")
    parser.add_argument("--retrack", action="store_true")
    args = parser.parse_args(split_prompt_args(sys.argv[1:]))

    raw = args.scans.resolve()
    if not raw.is_dir():
        sys.exit(f"{raw} is not a folder")
    store = ResultsStore(args.out or default_results_dir(raw), [raw])
    print(f"results: {store.root}")
    cfg = RunConfig(detector=args.detector, n_plates=args.plates, mask_fraction=args.mask_fraction,
                    threshold=args.threshold, tracker=args.tracker, workers=args.workers)

    if not args.review_only and not args.reflag and not args.retrack:
        scanners = find_scanners(raw)
        print(f"{len(scanners)} scanner(s): " + ", ".join(f"{s.name} ({len(s.scans)} scans)" for s in scanners))
        for s in scanners:
            done = store.root / s.name / "scanner.json"
            if done.exists() and not args.redo:
                prev = json.loads(done.read_text())
                print(f"{s.name}: already segmented with {prev['detector']} (use --redo to repeat)")
                continue
            if (store.root / s.name).exists():
                shutil.rmtree(store.path(s.name))
            plate_dirs = process_scanner(s, store, cfg, log=lambda m: print(m, flush=True))
            store.write_json({"detector": cfg.detector, "n_scans": len(s.scans), "plates": [d.name for d in plate_dirs],
                              "config": vars(cfg)}, s.name, "scanner.json")

    plate_dirs = load_plate_dirs(store.root)
    if not plate_dirs:
        sys.exit("no processed plates found")
    if args.retrack:
        for d in plate_dirs:
            r = retrack_plate(store, d, cfg.tracker)
            print(f"{d.parent.name}/{d.name}: {r['old_tracker']} -> {cfg.tracker} tracker, "
                  f"{r['old_colonies']} -> {r['colonies']} colonies, {r['flagged']} flagged", flush=True)
    elif args.reflag:
        for d in plate_dirs:
            print(f"{d.parent.name}/{d.name}: {refresh_flags(d)} flagged colonies")
    if args.no_review:
        export(store.root, store)
        return
    from scanlag.gui.review import run_review

    run_review(plate_dirs, on_finish=lambda: export(store.root, store))


if __name__ == "__main__":
    main()
