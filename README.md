



# GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs

#  Dependencies and Installation

We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create -n gp python=3.10
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# install SAM, LangSplat used SAM for multi-level segmentation. Here we using for part-level segmentation.
git clone https://github.com/minghanqin/segment-anything-langsplat
cd segment-anything; pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx 

pip install rembg
pip install numpy==1.26.4
pip install openai
pip install gradio
```


# 🧱 3D Reconstruction + Property Lifting



## Prerequisites
- Multi-view images under `gp_cases_dirs/<case>/images/` (e.g. `001.png`, `002.png`, ...)
- Run segmentation + VLM first (generates `seg/*_s.npy` and `<case>.txt`)
- Install COLMAP (ensure `colmap` is on PATH)
- 3D Gaussian Splatting training code is not vendored here. Use an external checkout (e.g. graphdeco-inria/gaussian-splatting) and point this repo to it.

## Run: COLMAP → 3DGS → Lift properties
Run inside your conda env (you mentioned `gp_sam2`):

```bash
# 1) segmentation + gpt_input
python sam_preprocess.py --dataset_path gp_cases_dirs

# 2) VLM labeling
python vlm_predict.py --dataset_path gp_cases_dirs --vlm qwen

# 3) reconstruct + lift (example case)
python reconstruction/pipeline.py \
  --case_dir gp_cases_dirs/doll \
  --gaussian_splatting_repo <PATH_TO_YOUR_GAUSSIAN_SPLATTING_REPO>
```

### Note: 3DGS requires undistorted COLMAP cameras

The official `graphdeco-inria/gaussian-splatting` loader only supports **undistorted** COLMAP datasets with camera models **PINHOLE** or **SIMPLE_PINHOLE**.
If your COLMAP model uses distortion (e.g. `SIMPLE_RADIAL`), undistort first:

- Script: [reconstruction/undistort_for_3dgs.py](reconstruction/undistort_for_3dgs.py)
- Example (lego): [scripts/undistort_lego_for_3dgs.cmd](scripts/undistort_lego_for_3dgs.cmd)

## End-to-end (Windows) quickstart: NeRF Synthetic `lego` → 3DGS → lift + colorize

This is the exact command sequence we’ve been running in this workspace.

### 0) Environments

We use two conda envs:

- `gp_sam2`: SAM2 segmentation + property lifting scripts
- `gs_cuda12`: official 3DGS training + CUDA extensions

### 1) Download a multi-view dataset (NeRF Synthetic)

This downloads NeRF Synthetic example data and copies a split into `gp_cases_dirs/<scene>/images/*.png`.

```bash
conda activate gp_sam2
python datasets/prepare_nerf_synthetic.py --scene lego --split train --out_root gp_cases_dirs --stride 1 --max_images 60 --force
```

### 2) COLMAP SfM (distorted model)

```bash
# Requires COLMAP installed and `colmap` on PATH
python reconstruction/run_colmap.py --images gp_cases_dirs/lego/images --workspace gp_cases_dirs/lego
```

### 3) Undistort for 3DGS (PINHOLE cameras)

```bash
# Run from an env that can execute our Python scripts (e.g., gp_sam2)
conda activate gp_sam2

# Writes gp_cases_dirs/lego_3dgs with undistorted images + sparse/0
scripts/undistort_lego_for_3dgs.cmd

# Convert undistorted model to TXT (needed by lifting)
colmap model_converter --input_path gp_cases_dirs/lego_3dgs/sparse/0 --output_path gp_cases_dirs/lego_3dgs/sparse_txt --output_type TXT
```

### 4) Build official 3DGS CUDA submodules (Windows)

We vendor the official repo under `external/gaussian-splatting/`.

```bash
conda activate gs_cuda12
scripts/build_gs_submodules.cmd
```

> If your Visual Studio / CUDA / conda paths differ, edit the variables at the top of `scripts/build_gs_submodules.cmd`.

### 5) Train 3DGS (7000 iters)

```bash
scripts/train_3dgs_lego_7000.cmd
```

Output Gaussians will be at:
- `Results_3dgs_lego_7000_cmd/point_cloud/iteration_7000/point_cloud.ply`

### 6) SAM2 segmentation on the UNDISTORTED images

```bash
conda activate gp_sam2
python sam_preprocess.py --dataset_path gp_cases_dirs --case_name lego_3dgs --model_type sam2 --sam_ckpt_path sam2_hiera_large.pt --model_cfg sam2_hiera_l.yaml
```

This creates:
- `gp_cases_dirs/lego_3dgs/seg/*_s.npy`
- `gp_cases_dirs/lego_3dgs/gpt_input/<view>/<part>.png`

### 7) Labels (choose ONE)

**Option A (real VLM):**

```bash
python vlm_predict.py --dataset_path gp_cases_dirs --case_name lego_3dgs --vlm qwen
```

**Option B (offline dummy labels for debugging):**

```bash
python reconstruction/generate_dummy_vlm_txt.py --case_dir gp_cases_dirs/lego_3dgs
```

### 8) Lift properties onto Gaussians (voting)

Run as a module from repo root so imports resolve:

```bash
python -m reconstruction.lift_properties_to_gaussians \
  --case_dir gp_cases_dirs/lego_3dgs \
  --colmap_model_txt gp_cases_dirs/lego_3dgs/sparse_txt \
  --gaussians_ply Results_3dgs_lego_7000_cmd/point_cloud/iteration_7000/point_cloud.ply \
  --out_ply Results_3dgs_lego_7000_cmd/point_cloud/iteration_7000/point_cloud_with_material_id.ply
```

### 9) Colorize the labeled Gaussians for viewing

```bash
python -m reconstruction.visualize_ply_materials \
  --ply Results_3dgs_lego_7000_cmd/point_cloud/iteration_7000/point_cloud_with_material_id.ply \
  --out_png Results_3dgs_lego_7000_cmd/point_cloud/iteration_7000/point_cloud_with_material_id.png \
  --out_colored_ply Results_3dgs_lego_7000_cmd/point_cloud/iteration_7000/point_cloud_with_material_id_colored.ply
```

The resulting colored PLY can be opened in CloudCompare/MeshLab:
- `Results_3dgs_lego_7000_cmd/point_cloud/iteration_7000/point_cloud_with_material_id_colored.ply`

Outputs go to `gp_cases_dirs/<case>/recon/` by default, including:
- `gaussians_with_material_id.ply`
- `gaussians_with_material_id.materials.txt` (material_id → material name)

## Run: Lift only (if you already have 3DGS + COLMAP)
```bash
python reconstruction/lift_properties_to_gaussians.py \
  --case_dir gp_cases_dirs/doll \
  --colmap_model_txt <COLMAP_TXT_MODEL_DIR> \
  --gaussians_ply <POINT_CLOUD_PLY> \
  --out_ply gp_cases_dirs/doll/recon/gaussians_with_material_id.ply
```





