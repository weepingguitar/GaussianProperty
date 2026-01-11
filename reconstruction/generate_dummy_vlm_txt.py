from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


_DEFAULT_MATERIALS = [
    "wood",
    "metal",
    "plastic",
    "glass",
    "fabric",
    "foam",
    "food",
    "ceramic",
    "paper",
    "leather",
]


def _shore_type(material: str) -> str:
    m = material.strip().lower()
    # Soft-ish materials -> Shore A, hard-ish -> Shore D (simple heuristic)
    if m in {"fabric", "foam", "leather"}:
        return "Shore A"
    return "Shore D"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate a VLM-style <case>.txt from SAM seg labels (offline). "
            "This is a cheap stand-in for vlm_predict.py so property lifting can run end-to-end."
        )
    )
    ap.add_argument("--case_dir", type=str, required=True, help="Case folder, e.g. gp_cases_dirs/lego")
    ap.add_argument(
        "--materials",
        type=str,
        default=",".join(_DEFAULT_MATERIALS),
        help="Comma-separated material library to cycle through",
    )
    ap.add_argument(
        "--out_txt",
        type=str,
        default="",
        help="Output txt path. Default: <case_dir>/<case_name>.txt",
    )
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    case_name = case_dir.name

    seg_dir = case_dir / "seg"
    if not seg_dir.exists():
        raise FileNotFoundError(f"Missing seg/ under: {case_dir}")

    materials: List[str] = [m.strip() for m in args.materials.split(",") if m.strip()]
    if not materials:
        raise ValueError("No materials provided")

    out_txt = Path(args.out_txt) if args.out_txt else (case_dir / f"{case_name}.txt")

    lines: List[str] = []
    seg_files = sorted(seg_dir.glob("*_s.npy"))
    for seg_path in seg_files:
        stem = seg_path.name.replace("_s.npy", "")
        try:
            view_id = int(stem)
        except ValueError:
            continue

        seg = np.load(seg_path)
        labels = sorted(int(x) for x in np.unique(seg) if int(x) >= 0)

        view_folder = str(view_id).zfill(2)

        for part_label in labels:
            # The repo's parser only reads the first 2 chars of the filename as the part index.
            # So we can only safely represent 0..99.
            if part_label < 0 or part_label > 99:
                continue

            part_file = f"{part_label:02d}.png"
            fake_image_path = str(case_dir / "gpt_input" / view_folder / part_file)

            material = materials[part_label % len(materials)]
            shore = _shore_type(material)
            hardness = "50-60" if shore == "Shore D" else "20-40"
            caption = f"part_{part_label}"

            # Required format: file_path, description, material, hardness, Shore A/D
            # Keep commas-free caption.
            lines.append(f"{fake_image_path},{caption},{material},{hardness},{shore}")

    if not lines:
        raise RuntimeError(f"No seg labels found under: {seg_dir}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote dummy VLM txt: {out_txt}")


if __name__ == "__main__":
    main()
