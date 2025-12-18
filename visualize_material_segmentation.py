import os
import argparse
import numpy as np
import matplotlib as mpl

from utils.vis_utils import parse_txt_file, filter_and_process, \
    filter_and_process, parse_material, make_legend, visualize_and_save_segmentation,cat_rgba

def vis_material_seg(base_path, vis_seg_save_base):
    case_list = os.listdir(base_path)
    cmap_tab10 = mpl.colormaps['tab10']

    for path in case_list:
        case_name = os.path.join(base_path, path)

        case_name_last = path
        image_base = os.path.join(case_name, "images")
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="gp_cases_dirs")
    args = parser.parse_args()

    base_path = args.dataset_path
    vis_seg_save_base = "./Results_" + os.path.basename(os.path.normpath(base_path))
    vis_material_seg(base_path, vis_seg_save_base)





        