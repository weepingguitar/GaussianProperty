from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a minimal COLMAP SfM pipeline and export TXT model")
    ap.add_argument("--images", type=str, required=True, help="Input images folder")
    ap.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Output workspace folder. For official 3DGS, this should usually be the scene root (same folder that contains images/).",
    )
    ap.add_argument("--single_camera", action="store_true", help="Assume all images share one camera")
    args = ap.parse_args()

    colmap = shutil.which("colmap")
    if not colmap:
        raise SystemExit(
            "COLMAP not found on PATH. Install COLMAP and ensure `colmap` is available in your terminal."
        )

    # Original behavior (relative paths depend on current working directory):
    # images = Path(args.images)
    # ws = Path(args.workspace)
    #
    # Safer behavior: resolve to absolute paths so running from subfolders works.
    images = Path(args.images).expanduser().resolve()
    ws = Path(args.workspace).expanduser().resolve()

    if not images.exists() or not images.is_dir():
        raise SystemExit(
            f"Images folder not found: {images}\n"
            f"Tip: pass an absolute path, or run from the repo root."
        )
    ws.mkdir(parents=True, exist_ok=True)

    # Standard COLMAP + 3DGS layout under workspace:
    #   database.db
    #   sparse/0
    #   sparse_txt/ (TXT export for our property lifting)
    database = ws / "database.db"
    sparse = ws / "sparse"
    sparse.mkdir(exist_ok=True)

    # Feature extraction
    cmd = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--ImageReader.single_camera",
        "1" if args.single_camera else "0",
    ]
    _run(cmd)

    # Matching
    cmd = [colmap, "exhaustive_matcher", "--database_path", str(database)]
    _run(cmd)

    # Mapping
    cmd = [
        colmap,
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse),
    ]
    _run(cmd)

    # Export TXT (we use sparse/0 as default)
    model0 = sparse / "0"
    txt_out = ws / "sparse_txt"
    txt_out.mkdir(exist_ok=True)

    cmd = [
        colmap,
        "model_converter",
        "--input_path",
        str(model0),
        "--output_path",
        str(txt_out),
        "--output_type",
        "TXT",
    ]
    _run(cmd)

    print(f"Wrote COLMAP TXT model to: {txt_out}")


def _run(cmd: list[str]) -> None:
    print(" ".join([str(c) for c in cmd]))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
