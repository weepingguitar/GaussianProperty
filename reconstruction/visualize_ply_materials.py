from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from reconstruction.ply_io import read_ply_vertices


def _load_material_names(materials_txt: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not materials_txt.exists():
        return mapping
    for line in materials_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        mapping[idx] = parts[1]
    return mapping


def _label_colors(labels: np.ndarray) -> np.ndarray:
    """Map integer labels to RGB uint8 colors."""
    import matplotlib as mpl

    labels = labels.astype(np.int32)
    uniq = np.unique(labels)

    # Use tab10; if >10 labels, cycle.
    cmap = mpl.colormaps["tab10"]
    lut: Dict[int, Tuple[int, int, int]] = {}
    for i, lab in enumerate(uniq.tolist()):
        if lab < 0:
            lut[lab] = (160, 160, 160)
        else:
            c = np.array(cmap(int(lab) % 10)[:3]) * 255
            lut[lab] = (int(c[0]), int(c[1]), int(c[2]))

    rgb = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for lab, col in lut.items():
        m = labels == lab
        rgb[m, 0] = col[0]
        rgb[m, 1] = col[1]
        rgb[m, 2] = col[2]
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize a PLY point cloud, coloring by material_id")
    ap.add_argument("--ply", type=str, required=True, help="Input PLY (e.g. sparse_points_with_material_id.ply)")
    ap.add_argument(
        "--materials_txt",
        type=str,
        default="",
        help="Optional materials mapping txt (default: <ply>.materials.txt if exists)",
    )
    ap.add_argument("--max_points", type=int, default=20000, help="Randomly subsample to this many points (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--point_size", type=float, default=1.0)
    ap.add_argument(
        "--clip_dist_percentile",
        type=float,
        default=99.5,
        help=(
            "Clip extreme outliers by distance-to-median percentile for visualization. "
            "Set to 0 to disable. Typical: 99-99.9"
        ),
    )
    ap.add_argument("--azim", type=float, default=45.0)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--out_png", type=str, default="", help="If set, save a PNG instead of only showing")
    ap.add_argument(
        "--out_colored_ply",
        type=str,
        default="",
        help="If set, write a copy of the PLY with RGB overwritten from material_id colors",
    )
    ap.add_argument("--show", action="store_true", help="Show interactive matplotlib window")
    args = ap.parse_args()

    ply_path = Path(args.ply)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    materials_txt = Path(args.materials_txt) if args.materials_txt else ply_path.with_suffix(".materials.txt")
    id_to_name = _load_material_names(materials_txt)

    ply = read_ply_vertices(str(ply_path))
    v = ply.vertex

    for f in ("x", "y", "z"):
        if f not in v.dtype.names:
            raise ValueError(f"PLY missing vertex field: {f}")
    if "material_id" not in v.dtype.names:
        raise ValueError("PLY missing vertex field: material_id")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    labels = v["material_id"].astype(np.int32)

    uniq, cnt = np.unique(labels, return_counts=True)
    print("label counts (material_id -> num_points):")
    for lab, c in zip(uniq.tolist(), cnt.tolist()):
        name = id_to_name.get(int(lab), "")
        suffix = f" ({name})" if name else ""
        print(f"  {lab}: {c}{suffix}")

    if args.clip_dist_percentile and args.clip_dist_percentile > 0:
        center = np.median(xyz, axis=0, keepdims=True)
        d = np.linalg.norm(xyz - center, axis=1)
        thr = float(np.percentile(d, args.clip_dist_percentile))
        keep = d <= thr
        if keep.sum() < xyz.shape[0]:
            print(
                f"clipping outliers: keep {int(keep.sum())}/{xyz.shape[0]} points (dist<=p{args.clip_dist_percentile}={thr:.3f})"
            )
            xyz = xyz[keep]
            labels = labels[keep]

    if args.max_points and args.max_points > 0 and xyz.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(xyz.shape[0], size=args.max_points, replace=False)
        xyz = xyz[idx]
        labels = labels[idx]

    # Colors must match the final xyz/labels arrays (after clipping/subsampling).
    rgb = _label_colors(labels)

    # Matplotlib 3D scatter
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=rgb / 255.0,
        s=args.point_size,
        depthshade=False,
        linewidths=0,
    )

    # Set equal-ish aspect
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title(f"{ply_path.name} (colored by material_id)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Simple legend text
    uniq = [int(x) for x in np.unique(labels) if int(x) >= 0]
    if uniq:
        legend_lines = []
        for lab in sorted(uniq)[:20]:
            name = id_to_name.get(lab, f"id_{lab}")
            legend_lines.append(f"{lab}: {name}")
        ax.text2D(0.02, 0.02, "\n".join(legend_lines), transform=ax.transAxes, fontsize=10)

    if args.out_png:
        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"Wrote: {out_png}")

    if args.out_colored_ply:
        # Write a copy of the PLY with RGB overwritten for easy viewing in external tools.
        out_ply = Path(args.out_colored_ply)
        out_ply.parent.mkdir(parents=True, exist_ok=True)
        out_v = v.copy()

        # If we clipped, we need to also clip vertex rows consistently.
        # When clipping is enabled, xyz/labels refer to a subset; in that case, write only that subset.
        if args.clip_dist_percentile and args.clip_dist_percentile > 0:
            center = np.median(
                np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64), axis=0, keepdims=True
            )
            d = np.linalg.norm(np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64) - center, axis=1)
            thr = float(np.percentile(d, args.clip_dist_percentile))
            keep = d <= thr
            out_v = out_v[keep]

        out_rgb = _label_colors(out_v["material_id"].astype(np.int32))

        # Reuse writer via ply_io.
        from reconstruction.ply_io import write_ply_vertices, append_vertex_fields

        if all(ch in out_v.dtype.names for ch in ("red", "green", "blue")):
            out_v["red"] = out_rgb[:, 0]
            out_v["green"] = out_rgb[:, 1]
            out_v["blue"] = out_rgb[:, 2]
        else:
            out_v = append_vertex_fields(
                out_v,
                new_fields={
                    "red": out_rgb[:, 0].astype(np.uint8),
                    "green": out_rgb[:, 1].astype(np.uint8),
                    "blue": out_rgb[:, 2].astype(np.uint8),
                },
                dtypes={"red": np.uint8, "green": np.uint8, "blue": np.uint8},
            )

        write_ply_vertices(str(out_ply), ply, out_v)
        print(f"Wrote: {out_ply}")

    # Previous behavior always popped an interactive window:
    # plt.show()
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
