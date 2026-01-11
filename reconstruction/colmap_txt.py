from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # model-specific

    def intrinsics_K(self) -> np.ndarray:
        """Return a 3x3 pinhole K for common COLMAP camera models.

        Supports: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL.
        Other models raise NotImplementedError.
        """
        m = self.model.upper()
        p = self.params.astype(np.float64)
        if m == "SIMPLE_PINHOLE":
            f, cx, cy = p
            fx = fy = f
        elif m == "PINHOLE":
            fx, fy, cx, cy = p
        elif m == "SIMPLE_RADIAL":
            f, cx, cy, _k = p
            fx = fy = f
        elif m == "RADIAL":
            f, cx, cy, _k1, _k2 = p
            fx = fy = f
        else:
            raise NotImplementedError(f"Unsupported camera model for K(): {self.model}")

        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return K


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str

    def qvec(self) -> np.ndarray:
        return np.array([self.qw, self.qx, self.qy, self.qz], dtype=np.float64)

    def tvec(self) -> np.ndarray:
        return np.array([self.tx, self.ty, self.tz], dtype=np.float64)

    def world_to_cam_Rt(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R, t) that maps X_world -> X_cam = R @ X_world + t."""
        R = qvec_to_rotmat(self.qvec())
        t = self.tvec()
        return R, t


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (qw,qx,qy,qz) to rotation matrix."""
    w, x, y, z = [float(v) for v in qvec]
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def load_colmap_model_txt(model_dir: str | Path) -> Tuple[Dict[int, ColmapCamera], Dict[int, ColmapImage]]:
    model_dir = Path(model_dir)
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"

    if not cameras_txt.exists() or not images_txt.exists():
        raise FileNotFoundError(
            f"Expected COLMAP TXT model with cameras.txt/images.txt under: {model_dir}"
        )

    cameras: Dict[int, ColmapCamera] = {}
    with cameras_txt.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]], dtype=np.float64)
            cameras[camera_id] = ColmapCamera(camera_id, model, width, height, params)

    images: Dict[int, ColmapImage] = {}
    with images_txt.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        images[image_id] = ColmapImage(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name)

        # COLMAP images.txt has a second line with 2D points; skip it.
        if i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
            i += 1

    return cameras, images
