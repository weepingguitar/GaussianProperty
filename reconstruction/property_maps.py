from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.vis_utils import parse_txt_file, filter_and_process, parse_material


@dataclass(frozen=True)
class ViewPropertyMaps:
    view_id: int  # integer view index (e.g., 1 for 001.png)
    seg_path: Path
    seg_labels: np.ndarray  # HxW int32 labels from *_s.npy
    part_to_material_id: Dict[int, int]  # seg label -> contiguous material id (0..M-1)

    def material_id_map(self, default_value: int = -1) -> np.ndarray:
        out = np.full_like(self.seg_labels, fill_value=default_value, dtype=np.int32)
        for part_label, mat_id in self.part_to_material_id.items():
            out[self.seg_labels == part_label] = int(mat_id)
        return out


def build_view_property_maps(case_dir: str | Path) -> Tuple[List[ViewPropertyMaps], List[str]]:
    """Build per-view material-id maps from this repo's outputs.

    Expects:
      - images/*.png
      - seg/*_s.npy
      - <case_name>.txt written by vlm_predict.py

    Returns:
      (views, material_names)

    material_names maps material_id -> material string.
    """
    case_dir = Path(case_dir)
    case_name = case_dir.name

    images_dir = case_dir / "images"
    seg_dir = case_dir / "seg"
    txt_path = case_dir / f"{case_name}.txt"

    if not images_dir.exists() or not seg_dir.exists():
        raise FileNotFoundError(f"Missing images/ or seg/ under: {case_dir}")
    if not txt_path.exists():
        raise FileNotFoundError(
            f"Missing VLM result txt: {txt_path}. Run vlm_predict.py first."
        )

    parsed = parse_txt_file(str(txt_path))

    # Determine available views by seg files.
    seg_files = sorted(seg_dir.glob("*_s.npy"))
    views: List[ViewPropertyMaps] = []

    # We build a global material-name list across views using parse_material per-view,
    # then unify by name.
    global_name_to_id: Dict[str, int] = {}
    global_id_to_name: List[str] = []

    for seg_path in seg_files:
        stem = seg_path.name.replace("_s.npy", "")
        try:
            view_id = int(stem)
        except ValueError:
            # Skip non-numeric views
            continue

        # From parsed_data, image_number is derived from gpt_input/XX/YY.png where XX is 2-digit view.
        # visualize_material_segmentation.py uses int(id) where id is e.g. "001" -> 1.
        materials_per_part = filter_and_process(parsed, view_id)
        # converted_list maps part_index -> local material id, but is view-local.
        converted_list, mat_names = parse_material(materials_per_part)

        # Build view-local part->global material id mapping.
        view_part_to_global: Dict[int, int] = {}
        for part_label, local_mat_id in enumerate(converted_list):
            if local_mat_id == -1:
                continue
            name = mat_names[local_mat_id]
            if name not in global_name_to_id:
                global_name_to_id[name] = len(global_id_to_name)
                global_id_to_name.append(name)
            view_part_to_global[part_label] = global_name_to_id[name]

        seg_labels = np.load(seg_path).astype(np.int32)
        views.append(
            ViewPropertyMaps(
                view_id=view_id,
                seg_path=seg_path,
                seg_labels=seg_labels,
                part_to_material_id=view_part_to_global,
            )
        )

    if not views:
        raise RuntimeError(f"No usable *_s.npy found under: {seg_dir}")

    return views, global_id_to_name
