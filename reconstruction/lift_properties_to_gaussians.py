from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from reconstruction.colmap_txt import load_colmap_model_txt
from reconstruction.ply_io import read_ply_vertices, write_ply_vertices, append_vertex_fields
from reconstruction.property_maps import build_view_property_maps


@dataclass(frozen=True)
class CameraView:
    view_id: int
    name: str
    K: np.ndarray  # 3x3
    R: np.ndarray  # 3x3 world->cam
    t: np.ndarray  # 3
    width: int
    height: int


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to pixel coordinates.

    Returns (u, v, z_cam)
    """
    Xc = (R @ xyz.T).T + t[None, :]
    z = Xc[:, 2]
    # Prevent division by zero
    valid = z > 1e-6

    u = np.full((xyz.shape[0],), np.nan, dtype=np.float64)
    v = np.full((xyz.shape[0],), np.nan, dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u[valid] = fx * (Xc[valid, 0] / z[valid]) + cx
    v[valid] = fy * (Xc[valid, 1] / z[valid]) + cy

    return u, v, z


def _vote_labels(labels: np.ndarray, num_classes: int) -> int:
    """labels: (N,) int with -1 meaning invalid."""
    valid = labels[labels >= 0]
    if valid.size == 0:
        return -1
    counts = np.bincount(valid, minlength=num_classes)
    return int(np.argmax(counts))


def main() -> None:
    ap = argparse.ArgumentParser(description="Lift 2D physical properties onto 3D Gaussians via frequency voting")
    ap.add_argument("--case_dir", type=str, required=True, help="Case folder under gp_cases_dirs/<case>")
    ap.add_argument("--colmap_model_txt", type=str, required=True, help="COLMAP TXT model folder containing cameras.txt/images.txt")
    ap.add_argument("--gaussians_ply", type=str, required=True, help="3DGS point_cloud.ply (binary PLY) to annotate")
    ap.add_argument("--out_ply", type=str, required=True, help="Output PLY with appended material_id field")
    ap.add_argument("--field_name", type=str, default="material_id", help="PLY field name to append")
    ap.add_argument("--max_gaussians", type=int, default=0, help="If >0, only process first N gaussians (debug)")
    args = ap.parse_args()

    case_dir = Path(args.case_dir)

    views, material_names = build_view_property_maps(case_dir)
    num_classes = max(1, len(material_names))

    cameras, images = load_colmap_model_txt(args.colmap_model_txt)

    # Map COLMAP image name -> CameraView. We assume COLMAP names match files in case images/.
    cam_views: Dict[int, CameraView] = {}
    for img_id, img in images.items():
        cam = cameras[img.camera_id]
        K = cam.intrinsics_K()
        R, t = img.world_to_cam_Rt()
        cam_views[img_id] = CameraView(
            view_id=_infer_view_id_from_name(img.name),
            name=img.name,
            K=K,
            R=R,
            t=t,
            width=cam.width,
            height=cam.height,
        )

    # Index property maps by view_id (e.g. 1,2,3...)
    viewid_to_map = {v.view_id: v.material_id_map() for v in views}

    ply = read_ply_vertices(args.gaussians_ply)
    vertex = ply.vertex

    if "x" not in vertex.dtype.names or "y" not in vertex.dtype.names or "z" not in vertex.dtype.names:
        raise ValueError("PLY vertex must contain x,y,z fields")

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    if args.max_gaussians and args.max_gaussians > 0:
        xyz = xyz[: args.max_gaussians]
        vertex = vertex[: args.max_gaussians]

    # Collect per-gaussian labels across views.
    per_view_labels: List[np.ndarray] = []

    # Iterate COLMAP images; only those with property maps.
    for _img_id, cv in cam_views.items():
        prop_map = viewid_to_map.get(cv.view_id)
        if prop_map is None:
            continue

        u, v, z = _project(cv.K, cv.R, cv.t, xyz)

        # Basic visibility: in front of camera and within bounds.
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        in_front = z > 1e-6
        in_bounds = (ui >= 0) & (ui < prop_map.shape[1]) & (vi >= 0) & (vi < prop_map.shape[0])
        ok = in_front & in_bounds

        labels = np.full((xyz.shape[0],), -1, dtype=np.int32)
        labels[ok] = prop_map[vi[ok], ui[ok]]
        per_view_labels.append(labels)

    if not per_view_labels:
        raise RuntimeError(
            "No overlapping COLMAP views and property maps. Ensure COLMAP image names correspond to case images and that seg/*.npy exist."
        )

    stacked = np.stack(per_view_labels, axis=1)  # (G, V)
    voted = np.full((xyz.shape[0],), -1, dtype=np.int32)
    for i in range(xyz.shape[0]):
        voted[i] = _vote_labels(stacked[i], num_classes=num_classes)

    out_vertex = append_vertex_fields(
        vertex,
        new_fields={args.field_name: voted.astype(np.int32)},
        dtypes={args.field_name: np.int32},
    )

    write_ply_vertices(args.out_ply, ply, out_vertex)

    # Also save material name mapping as a sidecar.
    mapping_path = Path(args.out_ply).with_suffix(".materials.txt")
    mapping_path.write_text("\n".join([f"{i}\t{name}" for i, name in enumerate(material_names)]), encoding="utf-8")

    print(f"Wrote: {args.out_ply}")
    print(f"Wrote: {mapping_path}")


def _infer_view_id_from_name(name: str) -> int:
    """Best-effort mapping from COLMAP image name to integer view id.

    Works for common naming like 001.png or 1.jpg. If parsing fails, returns -1.
    """
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        # try to extract trailing digits
        digits = "".join([c for c in stem if c.isdigit()])
        try:
            return int(digits) if digits else -1
        except ValueError:
            return -1


if __name__ == "__main__":
    main()
