from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Undistort a COLMAP sparse model into a 3DGS-compatible layout (PINHOLE/SIMPLE_PINHOLE).\n"
            "This runs `colmap image_undistorter` and then moves the produced sparse model into sparse/0/."
        )
    )
    ap.add_argument(
        "--scene_root",
        type=str,
        required=True,
        help="Input scene root that contains images/ and sparse/0 (COLMAP mapper output).",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root folder (will be created). It will contain images/ and sparse/0.",
    )
    ap.add_argument(
        "--colmap",
        type=str,
        default="",
        help="Optional path to COLMAP executable. If omitted, uses `colmap` from PATH.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output folder if it already exists (deletes its images/ and sparse/).",
    )
    args = ap.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    images = scene_root / "images"
    model0 = scene_root / "sparse" / "0"

    if not images.is_dir():
        raise SystemExit(f"Expected images/ folder at: {images}")
    if not model0.is_dir():
        raise SystemExit(f"Expected sparse/0 folder at: {model0}")

    colmap = args.colmap.strip() or shutil.which("colmap")
    if not colmap:
        raise SystemExit(
            "COLMAP executable not found. Provide --colmap or ensure `colmap` is on PATH."
        )

    if out_root.exists() and not args.force:
        raise SystemExit(
            f"Output folder already exists: {out_root}\n"
            "Pass --force to overwrite its images/ and sparse/ folders."
        )

    out_root.mkdir(parents=True, exist_ok=True)

    if args.force:
        for rel in ["images", "sparse"]:
            p = out_root / rel
            if p.exists():
                shutil.rmtree(p)

    cmd = [
        str(colmap),
        "image_undistorter",
        "--image_path",
        str(images),
        "--input_path",
        str(model0),
        "--output_path",
        str(out_root),
        "--output_type",
        "COLMAP",
    ]
    _run(cmd)

    # COLMAP writes to out_root/sparse/* (files directly inside sparse/)
    # 3DGS expects out_root/sparse/0/*.
    sparse_dir = out_root / "sparse"
    if not sparse_dir.is_dir():
        raise SystemExit(f"Expected COLMAP undistorter to create: {sparse_dir}")

    sparse0 = sparse_dir / "0"
    sparse0.mkdir(exist_ok=True)

    for child in list(sparse_dir.iterdir()):
        if child.name == "0":
            continue
        shutil.move(str(child), str(sparse0 / child.name))

    print(f"Undistorted dataset written to: {out_root}")


def _run(cmd: list[str]) -> None:
    print(" ".join([str(c) for c in cmd]))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
