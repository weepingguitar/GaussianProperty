from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Orchestrate 3D reconstruction + physical-property lifting")
    ap.add_argument("--case_dir", type=str, required=True, help="Case folder under gp_cases_dirs/<case>")
    ap.add_argument("--work_dir", type=str, default="", help="Working directory for reconstruction outputs")
    ap.add_argument("--colmap_single_camera", action="store_true")

    ap.add_argument(
        "--gaussian_splatting_repo",
        type=str,
        default="",
        help="Path to external gaussian-splatting repo (graphdeco-inria). Required to train 3DGS.",
    )
    ap.add_argument(
        "--gaussian_splatting_scene_dir",
        type=str,
        default="",
        help="Optional: override the scene folder passed to 3DGS training. Defaults to case_dir.",
    )

    ap.add_argument(
        "--gaussians_ply",
        type=str,
        default="",
        help="If provided, skip training and directly lift properties onto this PLY.",
    )
    ap.add_argument(
        "--colmap_model_txt",
        type=str,
        default="",
        help="If provided, skip COLMAP and use this TXT model folder.",
    )

    ap.add_argument("--out_ply", type=str, default="", help="Output labeled PLY path")
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.exists():
        raise SystemExit(f"case_dir not found: {case_dir}")

    work_dir = Path(args.work_dir) if args.work_dir else (case_dir / "recon")
    work_dir.mkdir(parents=True, exist_ok=True)

    images_dir = case_dir / "images"
    if not images_dir.exists():
        raise SystemExit(f"Missing images/ under: {case_dir}")

    # 1) COLMAP -> TXT model
    # Official 3DGS (graphdeco-inria/gaussian-splatting) expects COLMAP outputs under the same
    # scene directory passed via `-s`, typically:
    #   <scene>/images
    #   <scene>/sparse/0
    #   <scene>/database.db
    # So by default we run COLMAP with workspace=<case_dir>.
    #
    # Previous behavior (kept for reference):
    #   colmap_txt = (work_dir / "colmap" / "sparse_txt")
    #   workspace = (work_dir / "colmap")
    #
    colmap_txt = Path(args.colmap_model_txt) if args.colmap_model_txt else (case_dir / "sparse_txt")
    if not args.colmap_model_txt:
        _run_python(
            [
                str(Path(__file__).with_name("run_colmap.py")),
                "--images",
                str(images_dir),
                "--workspace",
                str(case_dir),
            ]
            + (["--single_camera"] if args.colmap_single_camera else [])
        )

    # 2) Train 3DGS (optional; external repo)
    gaussians_ply = Path(args.gaussians_ply) if args.gaussians_ply else None
    if gaussians_ply is None:
        if not args.gaussian_splatting_repo:
            raise SystemExit(
                "No --gaussians_ply provided and --gaussian_splatting_repo is empty. "
                "Provide a trained point_cloud.ply, or point to a gaussian-splatting checkout to train."
            )
        gs_repo = Path(args.gaussian_splatting_repo)
        if not gs_repo.exists():
            raise SystemExit(f"gaussian_splatting_repo not found: {gs_repo}")

        scene_dir = Path(args.gaussian_splatting_scene_dir) if args.gaussian_splatting_scene_dir else case_dir

        # NOTE: We don't vendor gaussian-splatting code here. This call assumes you have its deps installed.
        # You may need to adjust these args depending on the specific 3DGS repo fork you use.
        cmd = [
            "python",
            str(gs_repo / "train.py"),
            "-s",
            str(scene_dir),
            "--model_path",
            str(work_dir / "3dgs"),
        ]
        _run(cmd, cwd=str(gs_repo))

        # Best-effort: pick the latest point cloud.
        pc_root = work_dir / "3dgs" / "point_cloud"
        ply_candidates = sorted(pc_root.glob("iteration_*/point_cloud.ply"))
        if not ply_candidates:
            raise SystemExit(f"No point_cloud.ply found under: {pc_root}")
        gaussians_ply = ply_candidates[-1]

    # 3) Lift properties via voting
    out_ply = Path(args.out_ply) if args.out_ply else (work_dir / "gaussians_with_material_id.ply")
    _run_python(
        [
            str(Path(__file__).with_name("lift_properties_to_gaussians.py")),
            "--case_dir",
            str(case_dir),
            "--colmap_model_txt",
            str(colmap_txt),
            "--gaussians_ply",
            str(gaussians_ply),
            "--out_ply",
            str(out_ply),
        ]
    )

    print(f"Done. Labeled PLY: {out_ply}")


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print(" ".join([str(c) for c in cmd]))
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_python(args: list[str]) -> None:
    # Uses the current interpreter (so run inside gp_sam2)
    _run(["python", *args])


if __name__ == "__main__":
    main()
