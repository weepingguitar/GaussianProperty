import os
import cv2
import torch
import numpy as np
from PIL import Image
from rembg import remove
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils.sam_utils import *
from utils.vlm_utils import *
from utils.vis_utils import *
import os
import numpy as np
import matplotlib as mpl

base_path = "gradio_demo"
base_name = "demo"
sam_ckpt_path = "sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')


# Default material types
default_materials = "wood, metal, plastic, glass, fabric, foam, food, ceramic, paper, leather"

def update_materials(input_text):
    materials = input_text.split(",")
    materials = [material.strip() for material in materials]  # Remove any extra spaces

    return f"{', '.join(materials)}"



def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")
    os.system("rm -rf ./gradio_demo/*") 


def sam_image():

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=300,
    )

    # Process each dataset
    for dataset_id in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_id)
        img_folder = os.path.join(dataset_path, 'images')

        data_list = sorted(os.listdir(img_folder))

        img_list = []
        alpha_list = []

        for data_path in data_list:
            image_path = os.path.join(img_folder, data_path)
            image_rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            alpha = image_rgba[:, :, 3]

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



def preprocess(input_image, do_remove_background):

    # Create directories for saving processed images
    image_dir = os.path.join(base_path, base_name, 'images')
    os.makedirs(image_dir, exist_ok=True)

    if do_remove_background:
        # Use Rembg to remove the background and get the mask
        img_pil = remove(input_image)

    img_pil = resize_image(img_pil, 1280)

    # Save the processed image
    mask_save_path = os.path.join(image_dir, '001.png')
    img_pil.save(mask_save_path)

    seg_map_vis = sam_image()
    return seg_map_vis


def query_vlm(base_path, case_name, update_materials, vlm_type="qwen"):
    input_image_path = os.path.join(base_path, case_name, "gpt_input")
    image_files = get_image_files(input_image_path)
    if update_materials == '':
        update_materials = default_materials
    material_list = update_materials.split(", ")
    material_library = "{" + ", ".join(material_list) + "}"

    prompt = f"""Provided a picture. The left image is the original picture of the object (Original Image), and the middle image is a partial segmentation diagram (Mask Overlay), mask is in red. The right image is a partial of the object. 
    Based on the image, firstly provide a brief caption of the part. Secondly, describe what the part is made of (provide the major one). Finally, we combine what the object is and the material of the object to predict the hardness of the part. Choose whether to use Shore A hardness or Shore D hardness depending on the material. You may provide a range of values for hardness instead of a single value. 

    Format Requirement:
    You must provide your answer as a (brief caption of the part, material of the part, hardness, Shore A/D) pair. Do not include any other text in your answer, as it will be parsed by a code script later. 
    common material library: {material_library}. 
    Your answer must look like: caption, material, hardness low-high, <Shore A or Shore D>. 
    The material type must be chosen from the above common material library. Make sure to use Shore A or Shore D hardness, not Mohs hardness."""

    output_file = f'{case_name}.txt'
    results_file_path = os.path.join(base_path, case_name, output_file)

    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

    final_msg = ""
    with open(results_file_path, 'w') as file:
        for image_file in image_files:
            try:
                if vlm_type == 'qwen':
                    message = str(Qwen(image_file, prompt))
                else:
                    message = str(GPT4V(image_file, prompt))
            except KeyError as e:
                message = "error,-1"
            except Exception as e:
                message = "error,-1"
            write_msg = image_file + "," + message
            final_msg += message + "\n"
            file.write(f"{write_msg}\n")
            file.flush()

    print("Messages have been written to", results_file_path)
    return prompt, final_msg.rstrip("\n")


def run_vlm(unpdate_materials):
    print(unpdate_materials)
    save_gpt_input(base_path)

    all_cases = os.listdir(base_path)

    for case_name in all_cases:
        prompt,case_msg = query_vlm(base_path, case_name, unpdate_materials)
    return prompt, case_msg

def visualize_and_save_segmentation(original_image_path, mask_image_path, output_path, m_label, alpha=0.5):
    # Load the original image and mask
    original_image = np.array(Image.open(original_image_path).convert('RGBA'))
    mask_image = np.load(mask_image_path)  # Load mask as grayscale
    seg_map_vis = vis_segmap(mask_image)
    # Get the tab10 colormap
    cmap_tab10 = mpl.colormaps['tab10']

    # Create a color segmentation map with an alpha channel
    segmentation_map = np.zeros((mask_image.shape[0], mask_image.shape[1], 4), dtype=np.uint8)

    unique_labels = np.unique(mask_image)
    for label in unique_labels:
        if label == -1:
            segmentation_map[mask_image == label] = [0, 0, 0, 0]  # Transparent background
        else:
            color = np.array(cmap_tab10(m_label[label] % 10)[:3]) * 255
            segmentation_map[mask_image == label, :3] = color
            segmentation_map[mask_image == label, 3] = 255  # Fully opaque for labeled regions

    # Blend the original image with the segmentation map using the alpha parameter
    blended_image = np.zeros_like(original_image, dtype=np.uint8)
    for i in range(4):  # Loop through RGBA channels
        blended_image[:, :, i] = (original_image[:, :, i] * (1 - alpha) + 
                                  segmentation_map[:, :, i] * alpha).astype(np.uint8)

    # Create an empty image with an alpha channel for the combined image
    combined_image_with_alpha = np.zeros((original_image.shape[0], original_image.shape[1] * 2, 4), dtype=np.uint8)

    # Add the original image to the left side of the combined image
    combined_image_with_alpha[:, :original_image.shape[1], :] = original_image

    # Add the seg image to the right side of the combined image
    combined_image_with_alpha[:, original_image.shape[1]:2*original_image.shape[1], :] = blended_image

    # Save the combined image with transparency
    plt.imsave(output_path, combined_image_with_alpha)
    return Image.fromarray(combined_image_with_alpha)


def vis_material_seg(base_path, vis_seg_save_base):
    case_list = os.listdir(base_path)
    cmap_tab10 = mpl.colormaps['tab10']

    for path in case_list:
        case_name = os.path.join(base_path, path)

        case_name_last = path
        image_base = os.path.join(case_name, "images")
        vis_seg_base = os.path.join(case_name, "vis_seg")
        feature_base = os.path.join(case_name, "seg")
        number_view = len(os.listdir(image_base))

        gpt_txt = os.path.join(case_name, f"{case_name_last}.txt")
        parsed_data = parse_txt_file(gpt_txt)

        result_vis_seg_base = os.path.join(vis_seg_save_base, case_name_last)
        os.makedirs(result_vis_seg_base, exist_ok=True)

        for i in range(number_view):
            id = os.listdir(image_base)[i].split('.')[0]
            materials = filter_and_process(parsed_data, int(id))
            converted_list, mat_names = parse_material(materials)

            show = False
            legend = make_legend([cmap_tab10(i) for i in range(len(mat_names))], mat_names, 
                        savefile=os.path.join(result_vis_seg_base, f'{id}_legend.png'), show=show)

            img_path = os.path.join(image_base, id + '.png')
            s_path = os.path.join(feature_base, id + '_s.npy')
            ss = np.load(s_path)

            output_path = os.path.join(result_vis_seg_base, f"{id}_seg.png")
            vlm_result = visualize_and_save_segmentation(img_path, s_path, output_path, converted_list, alpha=0.9)

            print(case_name_last)
            final_result_vis = cat_rgba(vlm_result, legend)
            final_result_vis.save(os.path.join(result_vis_seg_base,"combined_image.png"))
    return final_result_vis


def save_vis_map():
    vis_seg_save_base = "./Results_" + os.path.basename(os.path.normpath(base_path))
    final_result_vis = vis_material_seg(base_path, vis_seg_save_base)
    return final_result_vis



import gradio as gr

_HEADER_ = """
    ## Material segmentation with [GaussianProperty](https://Gaussian-Property.github.io)
    * Upload an image of an object and click "Generate" to estimate the material. If the image has alpha channel, it be used as the mask. Otherwise, we use `rembg` to remove the background.
    * If you find that your material is not in the material candidate library, Enter materials (comma separated), click `Update Material Candidate Library`.
    * For convenience, we use Tongyi Qwen VL-Max-Latest here. You can use our [Code](https://Gaussian-Property.github.io) to perform inference with GPT-4V.
    """

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    width=256,
                    height=256,
                    type="pil",
                    elem_id="content_image",
                )
                seg_map_vis = gr.Image(
                    label="Segmentation by SAM", 
                    image_mode="RGB", 
                    width=256,
                    height=256,
                    type="pil", 
                    interactive=False
                )


            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
            with gr.Row():
                with gr.Group():
                    gr.Markdown("### Modify Material Types")
                    material_input = gr.Textbox(value=default_materials, label="Enter materials (comma separated)", lines=2)
                    unpdate_materials = gr.Textbox(label="Updated Materials", interactive=False)

                    # Button to trigger the update
                    update_button = gr.Button("Update Material Candidate Library")

            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")
            with gr.Row():

                gr.Examples(
                    examples=[
                        os.path.join("gp_cases", img_name) for img_name in sorted(os.listdir("gp_cases"))
                    ],
                    inputs=[input_image],
                    label="Examples",
                    cache_examples=False,
                    examples_per_page=3
                )


        with gr.Column():


            with gr.Row():
                with gr.Group():
                    
                    # Text box for displaying the System prompt and Agent Response
                    prompt = gr.Textbox(label="System Prompt",lines=12)
                    agent_msg = gr.Textbox(label="Agent Response", lines=5)


            with gr.Row():

                final_result_vis = gr.Image(
                    label="Input Image & Material Result & Legend",
                    type="pil",
                    width=512,
                    interactive=False
                )





    # Update output on button click
    update_button.click(fn=update_materials, inputs=[material_input], outputs=[unpdate_materials])

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[seg_map_vis],
    ).success(
        fn=run_vlm,
        inputs=[unpdate_materials],
        outputs=[prompt, agent_msg]
    ).success(
        fn=save_vis_map,
        inputs=[],
        outputs=[final_result_vis]
    )


demo.queue(max_size=10)
# demo.launch(server_name="127.0.0.1", server_port=8867)
demo.launch(server_name="127.0.0.1", server_port=7860)
