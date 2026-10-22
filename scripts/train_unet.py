"""Train the U-Net colony detector on all MATLAB test-set tiles plus the gold tiles.

    python scripts/train_unet.py [--name unet_scanlag_v1] [--steps 6000]

Uses ``<testset>/train/tiles.npz`` (built by scripts/train_cellpose.py) and writes
``models/<name>/model.pt``, ``config.json`` and ``history.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scanlag.store import PROJECT_ROOT, REGISTRY_FILE, ResultsStore
from scanlag.train.dataset import build_training_set, load_training_set
from scanlag.train.unet_train import UNetConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, default=PROJECT_ROOT / "testset_matlab")
    parser.add_argument("--name", default="unet_scanlag_v1")
    parser.add_argument("--steps", type=int, default=UNetConfig.steps)
    parser.add_argument("--scale", type=float, default=UNetConfig.scale)
    parser.add_argument("--min-upper-limit", type=float, default=0.0,
                        help="drop MATLAB tiles from plates whose contrast upper limit is below this "
                             "(low limits give noisy, oversized colony outlines)")
    args = parser.parse_args()

    raw_dirs = json.loads((args.testset / REGISTRY_FILE).read_text())["raw_dirs"]
    store = ResultsStore(args.testset, raw_dirs)
    tiles = args.testset / "train" / "tiles.npz"
    if not tiles.exists():
        tiles = build_training_set(args.testset, store, log=lambda m: print(m, flush=True))
    x, labels, ignore, meta = load_training_set(tiles)
    dropped = []
    if args.min_upper_limit > 0:
        upper = {pid: json.loads((args.testset / "plates" / pid / "plate.json").read_text())["limits"][1]
                 for pid in meta.plate_id.unique()}
        low = meta.plate_id.map(upper).lt(args.min_upper_limit) & (meta.source == "matlab")
        dropped = sorted(meta.plate_id[low].unique())
        print(f"dropping {int(low.sum())} tiles from {len(dropped)} plates with upper limit < {args.min_upper_limit}")
        keep = ~low.to_numpy()
        x, labels, ignore, meta = x[keep], labels[keep], ignore[keep], meta[keep].reset_index(drop=True)
    print(f"{len(x)} tiles ({(meta.source == 'gold').sum()} gold)", flush=True)

    models = ResultsStore(PROJECT_ROOT / "models", raw_dirs)
    cfg = UNetConfig(steps=args.steps, scale=args.scale)
    out = train(x, labels, ignore, meta.source.to_numpy(), models.path(args.name, "model.pt").parent, cfg,
                log=lambda m: print(m, flush=True))
    config_path = out.parent / "config.json"
    config = json.loads(config_path.read_text())
    config.update(min_upper_limit=args.min_upper_limit, dropped_plates=dropped)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
