from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen


NERF_EXAMPLE_DATA_URL = (
    "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip"
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download NeRF synthetic example data and convert a scene into gp_cases_dirs/<scene>/images/*.png"
    )
    ap.add_argument("--scene", type=str, default="lego", help="lego|drums|chair|hotdog|ficus|materials|mic|ship")
    ap.add_argument("--split", type=str, default="train", help="train|test|val")
    ap.add_argument("--out_root", type=str, default="gp_cases_dirs", help="Output root (will create <scene>/images)")
    ap.add_argument("--max_images", type=int, default=0, help="If >0, copy only first N images")
    ap.add_argument("--stride", type=int, default=1, help="Take every k-th image")
    ap.add_argument("--cache_dir", type=str, default="datasets_cache", help="Where to store downloaded zip/extracted data")
    ap.add_argument("--force", action="store_true", help="Overwrite existing output case folder")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "nerf_example_data.zip"
    extract_dir = cache_dir / "nerf_example_data"

    if not zip_path.exists():
        _download(NERF_EXAMPLE_DATA_URL, zip_path)

    if not extract_dir.exists():
        _extract(zip_path, extract_dir)

    # NeRF example zip contains: nerf_synthetic/<scene>/<split>/*.png
    src_images = extract_dir / "nerf_synthetic" / args.scene / args.split
    if not src_images.exists():
        raise SystemExit(f"Scene/split not found: {src_images}")

    out_case = Path(args.out_root) / args.scene
    out_images = out_case / "images"

    if out_case.exists() and args.force:
        shutil.rmtree(out_case)

    out_images.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in src_images.iterdir() if p.suffix.lower() == ".png"])
    if not images:
        raise SystemExit(f"No PNG images found in: {src_images}")

    images = images[:: max(1, int(args.stride))]
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    for i, p in enumerate(images, start=1):
        dst = out_images / f"{i:03d}.png"
        shutil.copy2(p, dst)

    print(f"Wrote {len(images)} images to: {out_images}")
    print("Next:")
    print(f"  conda activate gp_sam2")
    print(f"  python sam_preprocess.py --dataset_path {args.out_root} --model_type sam2 --sam_ckpt_path sam2_hiera_large.pt --model_cfg sam2_hiera_l.yaml")
    print(f"  python reconstruction/run_colmap.py --images {out_images} --workspace {out_case}")


def _download(url: str, dst: Path) -> None:
    print(f"Downloading: {url}")
    print(f"To: {dst}")
    with urlopen(url) as r:
        total = r.headers.get("Content-Length")
        total_size = int(total) if total else 0
        dst_tmp = dst.with_suffix(dst.suffix + ".partial")
        with dst_tmp.open("wb") as f:
            downloaded = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = 100.0 * downloaded / total_size
                    sys.stdout.write(f"\r  {downloaded/1e6:.1f}MB / {total_size/1e6:.1f}MB ({pct:.1f}%)")
                    sys.stdout.flush()
        if total_size:
            sys.stdout.write("\n")
    dst_tmp.replace(dst)


def _extract(zip_path: Path, extract_dir: Path) -> None:
    print(f"Extracting: {zip_path}")
    print(f"To: {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)


if __name__ == "__main__":
    main()
