"""Fine-tune Cellpose-SAM on all MATLAB test-set plates plus the gold tiles.

    python scripts/train_cellpose.py [--name cpsam_scanlag_v1] [--steps 8000] [--rebuild-tiles]

1. Builds ``<testset>/train/tiles.npz`` once (about 3k tiles read from the raw scans).
2. Trains and writes ``models/<name>/model.pt``, ``config.json`` and ``history.json``.

All plates and gold tiles are used for training, so scores on the gold set afterwards only show
that the model fits its training data. A real test needs new, unseen plates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scanlag.store import PROJECT_ROOT, REGISTRY_FILE, ResultsStore
from scanlag.train.cellpose_finetune import TrainConfig, train
from scanlag.train.dataset import build_training_set, load_training_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, default=PROJECT_ROOT / "testset_matlab")
    parser.add_argument("--name", default="cpsam_scanlag_v1")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--accumulate", type=int, default=TrainConfig.accumulate)
    parser.add_argument("--rebuild-tiles", action="store_true")
    args = parser.parse_args()

    raw_dirs = json.loads((args.testset / REGISTRY_FILE).read_text())["raw_dirs"]
    store = ResultsStore(args.testset, raw_dirs)
    tiles = args.testset / "train" / "tiles.npz"
    if args.rebuild_tiles or not tiles.exists():
        tiles = build_training_set(args.testset, store, log=lambda m: print(m, flush=True))
    x, labels, ignore, meta = load_training_set(tiles)
    print(f"{len(x)} tiles ({(meta.source == 'gold').sum()} gold)", flush=True)

    models = ResultsStore(PROJECT_ROOT / "models", raw_dirs)
    cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, accumulate=args.accumulate)
    out = train(x, labels, ignore, meta.source.to_numpy(), models.path(args.name, "model.pt").parent, cfg,
                log=lambda m: print(m, flush=True))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
