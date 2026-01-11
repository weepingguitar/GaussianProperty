import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Toggle to enable/disable post-filtering/merging of masks
MASK_FILTER_ENABLED = False

def resize_image(image, max_size=1280):
    # Get the current size of the image
    width, height = image.size

    # Determine the longest side
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int((max_size / width) * height)
            image = image.resize((new_width, new_height))
    else:
        if height > max_size:
            new_height = max_size
            new_width = int((max_size / height) * width)
            image = image.resize((new_width, new_height))

    return image


def seed_everything(seed_value):
    """Seeds all random number generators for reproducibility."""
    import torch
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def create(image_list, alpha_list, data_list, save_folder, mask_generator):
    """Generates segmentation maps for each image in the list."""
    assert image_list is not None, "image_list must be provided to generate features"
    mask_generator.predictor.model.to('cuda')
    for i, img in tqdm(enumerate(image_list)):
        alpha = alpha_list[i]
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        seg_map_vis = sam_encoder(img.unsqueeze(0), alpha, save_path, mask_generator)
    
    return seg_map_vis


def save_numpy(save_path, seg_map):
    """Saves segmentation maps as numpy files."""
    save_path_s = save_path + '_s.npy'
    np.save(save_path_s, seg_map)


def get_seg_img(image, mask, bbox):
    """Extracts a segmented image using the mask and bounding box."""
    image = image.copy()
    image[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(bbox)
    return image[y:y+h, x:x+w, ...]


def pad_img(img):
    """Pads the image to make it square."""
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad


def filter(keep, masks_result) -> list:
    """Filters masks based on the indices in `keep`."""
    import torch
    keep = keep.int().cpu().numpy()
    return [m for i, m in enumerate(masks_result) if i in keep]


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2):
    """Performs non-maximum suppression on masks."""
    import torch
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True

    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    return idx[keep]


def masks_update(*args, **kwargs):
    """Removes redundant masks based on scores and overlap rate."""
    import torch
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new


def get_location(image, foreground_mask):
    """Finds the bounding box for the largest contour in the mask."""
    contours, _ = cv2.findContours(foreground_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    return [x, y, w, h]


def vis_segmap_sam(seg_map, debug_vis_path):
    """Visualizes the segmentation map."""
    vis_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    unique_values = np.unique(seg_map)
    colors = {value: [255, 255, 255] if value == -1 else list(np.random.choice(range(256), size=3)) for value in unique_values}
    for key, color in colors.items():
        vis_mask[seg_map == key] = color

    cv2.imwrite(f'{os.path.join(debug_vis_path, "seg_map.png")}', vis_mask[:, :, [2, 1, 0]])
    return vis_mask[:, :, [2, 1, 0]]


def is_valid_mask(mask_data):
    """
    Advanced filtering for SAM masks to remove noise.
    Checks: Area, Stability Score, Predicted IoU, and Shape Solidity.
    """
    # 1. Area Threshold (increased)
    if mask_data['area'] < 500:
        return False

    # 2. Stability Score (SAM metric for mask consistency)
    # Higher threshold ensures we only keep very stable masks
    if mask_data.get('stability_score', 1.0) < 0.88:
        return False

    # 3. Predicted IoU (SAM confidence)
    if mask_data.get('predicted_iou', 1.0) < 0.85:
        return False

    # 4. Shape Analysis: Solidity
    # Solidity = Contour Area / Convex Hull Area
    # Noise tends to be jagged, irregular, or stringy (low solidity).
    # Real object parts tend to be more convex/solid.
    mask_uint8 = mask_data['segmentation'].astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False
    
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    
    if hull_area > 0:
        solidity = float(mask_data['area']) / hull_area
        if solidity < 0.6: # Filter out very jagged/irregular shapes
            return False
            
    return True


def sam_encoder(image, alpha, save_path, mask_generator):
    """Encodes the image and generates segmentation maps."""
    vis_seg_path = save_path.replace("seg", "vis_seg")

    # Clean previous artifacts so stale masks don't persist between runs
    part_dir = os.path.join(vis_seg_path, "part")
    if os.path.exists(part_dir):
        for f in os.listdir(part_dir):
            try:
                os.remove(os.path.join(part_dir, f))
            except OSError:
                pass
    else:
        os.makedirs(part_dir, exist_ok=True)
    os.makedirs(vis_seg_path, exist_ok=True)

    image = cv2.cvtColor(image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    # Generate masks
    masks_output = mask_generator.generate(image)

    # Handle different SAM versions/outputs
    if isinstance(masks_output, tuple) and len(masks_output) == 4:
        # Legacy/Custom SAM returning 4 values
        masks_default, masks_s, masks_m, masks_l = masks_output
    else:
        # Standard SAM 1 / SAM 2 returning a single list
        masks_m = masks_output

    masks_m = masks_update(masks_m, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)[0]

    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    seg_map[alpha == 255] = 0

    masks_m = sorted(masks_m, key=lambda x: x['area'], reverse=True)

    for kk, mask in enumerate(masks_m):
        if kk == 0: # Skip the largest mask (usually background)
            continue
            
        if MASK_FILTER_ENABLED:
            # Apply advanced filtering
            if not is_valid_mask(mask):
                continue
        # When disabled, accept all masks
        seg_map[mask['segmentation']] = kk

    seg_map[alpha == 0] = -1

    if MASK_FILTER_ENABLED:
        # --- CLEANUP: Merge small noise from Label 0 into nearest neighbors ---
        label_0_mask = (seg_map == 0).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(label_0_mask, connectivity=8)
        seg_map_refined = seg_map.copy()
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 300:
                component_mask = (labels_im == i).astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                dilated_mask = cv2.dilate(component_mask, kernel, iterations=1)
                boundary_mask = dilated_mask - component_mask
                neighbor_labels = seg_map[boundary_mask == 1]
                valid_neighbors = neighbor_labels[(neighbor_labels != -1) & (neighbor_labels != 0)]
                if len(valid_neighbors) > 0:
                    counts = np.bincount(valid_neighbors)
                    most_frequent_label = np.argmax(counts)
                    seg_map_refined[labels_im == i] = most_frequent_label
                else:
                    seg_map_refined[labels_im == i] = -1
        seg_map = seg_map_refined

        # --- CLEANUP STEP 2: Filter Label 0 (Remainder) for Shape Quality ---
        label_0_mask = (seg_map == 0).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(label_0_mask, connectivity=8)
        for i in range(1, num_labels):
            component_mask = (labels_im == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(cnt)
                if hull_area > 0:
                    solidity = contour_area / hull_area
                    if solidity < 0.2:
                        component_mask = (labels_im == i).astype(np.uint8)
                        kernel = np.ones((3,3), np.uint8)
                        dilated_mask = cv2.dilate(component_mask, kernel, iterations=1)
                        boundary_mask = dilated_mask - component_mask
                        neighbor_labels = seg_map[boundary_mask == 1]
                        valid_neighbors = neighbor_labels[(neighbor_labels != -1) & (neighbor_labels != 0)]
                        if len(valid_neighbors) > 0:
                            counts = np.bincount(valid_neighbors)
                            most_frequent_label = np.argmax(counts)
                            seg_map[labels_im == i] = most_frequent_label
                        else:
                            seg_map[labels_im == i] = -1

        # --- FINAL SAFETY CHECK: Merge any label (including 0) that is too small ---
        unique_labels = np.unique(seg_map)
        for label in unique_labels:
            if label == -1:
                continue
            if np.sum(seg_map == label) < 500:
                component_mask = (seg_map == label).astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                dilated_mask = cv2.dilate(component_mask, kernel, iterations=1)
                boundary_mask = dilated_mask - component_mask
                neighbor_labels = seg_map[boundary_mask == 1]
                valid_neighbors = neighbor_labels[(neighbor_labels != -1) & (neighbor_labels != label)]
                if len(valid_neighbors) > 0:
                    counts = np.bincount(valid_neighbors)
                    most_frequent_label = np.argmax(counts)
                    seg_map[seg_map == label] = most_frequent_label
                else:
                    seg_map[seg_map == label] = -1

    seg_map_vis = vis_segmap_sam(seg_map, vis_seg_path)

    for i in np.unique(seg_map):
        if i == -1:
            continue
        cur_mask = seg_map == i
        bbox = get_location(image, cur_mask)
        seg_img = get_seg_img(image, cur_mask, bbox)
        pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
        cv2.imwrite(f"{vis_seg_path}/part/mask_{i}.png", pad_seg_img[:, :, [2, 1, 0]])

    save_numpy(save_path, seg_map)
    return Image.fromarray(seg_map_vis)


def save_gpt_input(base_path, case_name: str = ""):
    all_cases = os.listdir(base_path)
    if case_name:
        all_cases = [case_name]

    for path in all_cases:
        case_name = os.path.join(base_path, path)

        image_base = f"{case_name}/images"
        # NOTE: original code assumes a single view:
        # number_view = 1
        # For 3D reconstruction + multi-view voting, we must support multi-view inputs.
        # We interpret each image in `images/` as a view.
        if not os.path.exists(image_base):
            continue
        image_files = sorted([f for f in os.listdir(image_base) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        number_view = len(image_files)
        feature_base = f"{case_name}/seg"
        vis_seg_base = f"{case_name}/vis_seg"

        base_gpt_test_path = os.path.join(case_name, "gpt_input")
        os.makedirs(base_gpt_test_path, exist_ok=True)

        for i in range(1, number_view + 1):
            cur_gpt_path = os.path.join(base_gpt_test_path, str(i).zfill(2))
            os.makedirs(cur_gpt_path, exist_ok=True)

            # Map the i-th view to its filename (supports arbitrary naming as long as it is numeric-sortable).
            # Fall back to the old convention (001.png) if the file list is empty.
            if number_view > 0:
                view_stem = os.path.splitext(image_files[i - 1])[0]
            else:
                view_stem = str(i).zfill(3)

            img_path = os.path.join(image_base, f"{view_stem}.png")
            if not os.path.exists(img_path):
                # Try original extension if not PNG.
                img_path = os.path.join(image_base, image_files[i - 1])
            s_path = os.path.join(feature_base, f"{view_stem}_s.npy")
            seg_path = os.path.join(vis_seg_base, f"{view_stem}/part")
            ss = np.load(s_path)
            rgba_image = cv2.imread(img_path)
            image = cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGB)

            mask = ss

            # Find different labels in the mask
            labels = np.unique(mask)
            labels = labels[labels != -1]  # Remove non-material parts from the mask

            # Generate random colors for each label
            colors = {}
            for label in labels:
                colors[label] = (random.random(), random.random(), random.random())

            # Create color mapping
            cmap = ListedColormap([colors[label] for label in labels])

            for label in labels:
                # Filter out small masks to avoid sending noise to VLM
                if np.sum(mask == label) < 300:
                    continue

                part_image_path = os.path.join(seg_path, f"mask_{label}.png")
                if not os.path.exists(part_image_path):
                    continue
                    
                part_image = cv2.imread(part_image_path)
                part_image = cv2.cvtColor(part_image, cv2.COLOR_BGR2RGB)

                # Create plot
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))  # 1 row, 3 columns

                # Display original image on the left
                ax1.imshow(image)
                ax1.set_title('Original Image')
                ax1.axis('off')  # Turn off axis

                # Display mask overlay on the middle
                ax2.imshow(image)
                masked_image = np.ma.masked_where(mask != label, mask)
                ax2.imshow(masked_image, cmap=cmap, alpha=0.4, vmin=np.min(mask), vmax=np.max(mask))
                ax2.set_title('Mask Overlay')
                ax2.axis('off')

                # Display part image on the right
                ax3.imshow(part_image)
                ax3.set_title('Part Image')
                ax3.axis('off')

                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

                # Save the image to file
                plt.savefig(f'{cur_gpt_path}/{str(label).zfill(2)}.png')

                plt.close()
