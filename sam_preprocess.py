import os
import cv2
import torch
import argparse
import numpy as np
from utils.sam_utils import create, seed_everything, save_gpt_input

def sam_image(mask_generator, base_path, case_name: str = ""):
    seg_map_vis = None
    # Process each dataset
    all_cases = os.listdir(base_path)
    if case_name:
        # Only process one case (keeps original all-cases behavior available)
        all_cases = [case_name]

    for dataset_id in all_cases:
        dataset_path = os.path.join(base_path, dataset_id)
        img_folder = os.path.join(dataset_path, 'images')

        if not os.path.exists(img_folder):
            continue

        data_list = sorted(os.listdir(img_folder))

        img_list = []
        alpha_list = []

        for data_path in data_list:
            image_path = os.path.join(img_folder, data_path)
            image_rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            
            if image_rgba.shape[2] == 4:
                alpha = image_rgba[:, :, 3]
            else:
                alpha = np.ones(image_rgba.shape[:2], dtype=np.uint8) * 255

            # Ensure alpha mask is binary
            alpha[alpha < 125] = 0
            alpha[alpha >= 125] = 255

            image = cv2.imread(image_path)
            image = torch.from_numpy(image)

            img_list.append(image)
            alpha_list.append(alpha[None, ...])

        # Prepare images and alphas for processing
        images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
        imgs = torch.cat(images)
        alphas = np.concatenate(alpha_list, 0)

        save_folder = os.path.join(dataset_path, 'seg')
        os.makedirs(save_folder, exist_ok=True)

        # Generate segmentation maps
        seg_map_vis = create(imgs, alphas, data_list, save_folder, mask_generator)
    
    return seg_map_vis

if __name__ == '__main__':
    seed_everything(42)

    parser = argparse.ArgumentParser(description = "Part-level Segmentation using SAM")
    parser.add_argument('--dataset_path', type=str, default="gp_cases_dirs")
    parser.add_argument('--case_name', type=str, default="", help="If set, only process this subfolder under dataset_path")
    parser.add_argument('--sam_ckpt_path', type=str, default="./sam_vit_h_4b8939.pth")
    parser.add_argument('--model_type', type=str, default="vit_h", help="vit_h, vit_b, sam2")
    parser.add_argument('--model_cfg', type=str, default="sam2_hiera_l.yaml", help="Config for SAM2")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    base_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    
    if args.model_type == "sam2":
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            
            print(f"Loading SAM2 model from {sam_ckpt_path}...")
            
            # Custom build logic to handle local config files reliably
            if os.path.exists(args.model_cfg):
                print(f"Using local config file: {args.model_cfg}")
                cfg = OmegaConf.load(args.model_cfg)
                sam_model = instantiate(cfg.model, _recursive_=True)
                
                # Load checkpoint
                if sam_ckpt_path:
                    sd = torch.load(sam_ckpt_path, map_location="cpu")["model"]
                    sam_model.load_state_dict(sd)
                
                sam_model = sam_model.to('cuda')
                sam_model.eval()
            else:
                # Fallback to library function
                sam_model = build_sam2(args.model_cfg, sam_ckpt_path, device='cuda', apply_postprocessing=False)

            mask_generator = SAM2AutomaticMaskGenerator(
                model=sam_model,
                points_per_side=32,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.92,
                box_nms_thresh=0.7,
                min_mask_region_area=300,
            )
        except ImportError as e:
            print(f"Error: SAM2 not installed or configured correctly. {e}")
            exit(1)
        except Exception as e:
            print(f"Error loading SAM2: {e}")
            exit(1)
    else:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        
        print(f"Loading SAM1 ({args.model_type}) model from {sam_ckpt_path}...")
        sam_model = sam_model_registry[args.model_type](checkpoint=sam_ckpt_path).to('cuda')
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=300,
        )

    # IMPORTANT:
    # Do NOT rewrite base_path to base_path/case_name here.
    # That would make sam_image() look for <case>/<case>/images (e.g. lego_3dgs/lego_3dgs/images) and do nothing.
    # Single-case behavior is implemented by filtering inside sam_image() and save_gpt_input().
    #
    # if args.case_name:
    #     base_path = os.path.join(base_path, args.case_name)

    # Previous attempt (do not use):
    # if args.case_name:
    #     base_path = os.path.join(base_path, args.case_name)

    sam_image(mask_generator, base_path, case_name=args.case_name)
    save_gpt_input(base_path, case_name=args.case_name)

    
