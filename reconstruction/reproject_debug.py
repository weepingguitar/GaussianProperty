from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from reconstruction.colmap_txt import load_colmap_model_txt
from reconstruction.ply_io import read_ply_vertices


def _infer_image_name(view_id: int) -> str:
    return f"{view_id:03d}.png"


def _project_points(
    cam_model: str,
    cam_params: np.ndarray,
    width: int,
    height: int,
    R: np.ndarray,
    t: np.ndarray,
    xyz_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project points with basic COLMAP distortion for common models.

    Returns (u, v, z_cam, valid_mask)
    """
    cam_model = cam_model.upper()
    p = cam_params.astype(np.float64)

    Xc = (R @ xyz_world.T).T + t[None, :]
    z = Xc[:, 2]
    in_front = z > 1e-6

    # normalized coords
    x = np.zeros_like(z, dtype=np.float64)
    y = np.zeros_like(z, dtype=np.float64)
    x[in_front] = Xc[in_front, 0] / z[in_front]
    y[in_front] = Xc[in_front, 1] / z[in_front]

    # Apply distortion if present (approx COLMAP models)
    if cam_model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        fx = fy = f
    elif cam_model == "PINHOLE":
        fx, fy, cx, cy = p
    elif cam_model == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        fx = fy = f
        r2 = x * x + y * y
        radial = 1.0 + k * r2
        x = x * radial
        y = y * radial
    elif cam_model == "RADIAL":
        f, cx, cy, k1, k2 = p
        fx = fy = f
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * (r2 * r2)
        x = x * radial
        y = y * radial
    else:
        # Fallback: treat as pinhole using first params like SIMPLE_RADIAL (best effort)
        # This keeps the debug script usable even if COLMAP chooses a different model.
        f = float(p[0]) if p.size > 0 else 1.0
        cx = float(p[1]) if p.size > 1 else width / 2.0
        cy = float(p[2]) if p.size > 2 else height / 2.0
        fx = fy = f

    u = fx * x + cx
    v = fy * y + cy

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid = in_front & in_bounds & np.isfinite(u) & np.isfinite(v)

    return u, v, z, valid


def _color_from_label(label: int) -> Tuple[int, int, int]:
    # BGR for OpenCV
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    if label < 0:
        return (180, 180, 180)
    c = palette[int(label) % len(palette)]
    return (c[2], c[1], c[0])


def _render_seg_viz(seg: np.ndarray) -> np.ndarray:
    # Deterministic visualization: label -> tab10-ish colors, background transparent-ish
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    uniq = np.unique(seg)
    for lab in uniq.tolist():
        if int(lab) < 0:
            continue
        color = _color_from_label(int(lab))
        out[seg == lab] = color
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproject 3D points into an image to verify camera alignment")
    ap.add_argument("--case_dir", type=str, required=True)
    ap.add_argument("--colmap_model_txt", type=str, required=True)
    ap.add_argument("--points_ply", type=str, required=True)
    ap.add_argument("--view_id", type=int, default=1, help="View index, e.g. 1 for 001.png")
    ap.add_argument("--num_points", type=int, default=3000, help="Randomly sample this many points (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=2, help="Circle radius in pixels")
    ap.add_argument("--alpha_seg", type=float, default=0.35, help="Alpha for seg overlay")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory (default: <case_dir>/reproj_debug)")
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    images_dir = case_dir / "images"
    seg_dir = case_dir / "seg"

    image_name = _infer_image_name(args.view_id)
    image_path = images_dir / image_name
    seg_path = seg_dir / f"{args.view_id:03d}_s.npy"

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing seg: {seg_path}")

    cameras, images = load_colmap_model_txt(args.colmap_model_txt)

    # Find the COLMAP image record by name (exact match) or stem match.
    colmap_img = None
    for _img_id, img in images.items():
        if img.name == image_name:
            colmap_img = img
            break
    if colmap_img is None:
        for _img_id, img in images.items():
            if Path(img.name).stem == Path(image_name).stem:
                colmap_img = img
                break
    if colmap_img is None:
        raise RuntimeError(f"Could not find COLMAP image entry for {image_name}")

    cam = cameras[colmap_img.camera_id]
    R, t = colmap_img.world_to_cam_Rt()

    ply = read_ply_vertices(args.points_ply)
    v = ply.vertex
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    labels: Optional[np.ndarray] = None
    if "material_id" in v.dtype.names:
        labels = v["material_id"].astype(np.int32)

    if args.num_points and args.num_points > 0 and xyz.shape[0] > args.num_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(xyz.shape[0], size=args.num_points, replace=False)
        xyz = xyz[idx]
        if labels is not None:
            labels = labels[idx]

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"cv2 failed to read image: {image_path}")

    seg = np.load(seg_path).astype(np.int32)

    u, v_pix, z, valid = _project_points(
        cam.model,
        cam.params,
        cam.width,
        cam.height,
        R,
        t,
        xyz,
    )

    ui = np.round(u[valid]).astype(np.int32)
    vi = np.round(v_pix[valid]).astype(np.int32)

    # Compute strict-ish metric: how many projected points land on foreground seg (seg>=0)
    on_fg = seg[vi, ui] >= 0
    fg_ratio = float(on_fg.mean()) if on_fg.size else 0.0

    print("reprojection stats:")
    print(f"  view: {image_name}")
    print(f"  points total: {xyz.shape[0]}")
    print(f"  projected in-bounds: {int(valid.sum())}")
    print(f"  on segmentation foreground: {fg_ratio*100:.2f}%")

    # Render overlay on original image
    overlay = img_bgr.copy()
    if labels is None:
        for x, y in zip(ui.tolist(), vi.tolist()):
            cv2.circle(overlay, (x, y), args.radius, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    else:
        lab_valid = labels[valid]
        for (x, y, lab) in zip(ui.tolist(), vi.tolist(), lab_valid.tolist()):
            cv2.circle(overlay, (x, y), args.radius, _color_from_label(int(lab)), thickness=-1, lineType=cv2.LINE_AA)

    # Render segmentation overlay + points
    seg_viz = _render_seg_viz(seg)
    seg_on_img = cv2.addWeighted(img_bgr, 1.0 - args.alpha_seg, seg_viz, args.alpha_seg, 0.0)
    seg_overlay = seg_on_img

    if labels is None:
        for x, y in zip(ui.tolist(), vi.tolist()):
            cv2.circle(seg_overlay, (x, y), args.radius, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    else:
        lab_valid = labels[valid]
        for (x, y, lab) in zip(ui.tolist(), vi.tolist(), lab_valid.tolist()):
            cv2.circle(seg_overlay, (x, y), args.radius, _color_from_label(int(lab)), thickness=-1, lineType=cv2.LINE_AA)

    out_dir = Path(args.out_dir) if args.out_dir else (case_dir / "reproj_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    out1 = out_dir / f"view_{args.view_id:03d}_reproj_on_image.png"
    out2 = out_dir / f"view_{args.view_id:03d}_reproj_on_seg.png"

    cv2.imwrite(str(out1), overlay)
    cv2.imwrite(str(out2), seg_overlay)

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")


if __name__ == "__main__":
    main()
