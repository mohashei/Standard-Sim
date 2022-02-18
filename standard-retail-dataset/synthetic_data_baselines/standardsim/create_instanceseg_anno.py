"""
Contains functions to generate labels for instance segmentation
"""
import cv2
import os
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import ConvexHull
from shapely import geometry
import skimage
import skimage.morphology


def get_category_dict():
    """ Given categories list from annotation json,
    returns a mapping from category id to name
    """
    raise NotImplementedError

def get_annotations_from_image_id():
    """ Given an image id, returns list of 
    """
    raise NotImplementedError

def make_cammogram(im1_path, label_file, segmentation_file, im1_depth):
    segmentation_file = im1_path.replace(".png", "-segmentation0001.exr")
    data = json.load(open(label_file, "r"))
    sku_name_to_section_name = data["sku_name_to_section_name"]
    index_mapping = data["index_mapping"]
    segment_id_to_section_name = {index_mapping[k]:v for k,v in sku_name_to_section_name.items()}
    segmentation_image = cv2.imread(segmentation_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
    
    ids_in_image = np.unique(segmentation_image)
    section_masks = defaultdict(lambda: np.array(np.zeros((segmentation_image.shape[0], segmentation_image.shape[1]), int)))
    for segment_id in ids_in_image:
        if segment_id not in segment_id_to_section_name:
            continue
        section_name = segment_id_to_section_name[int(segment_id)]
        section_mask = section_masks[section_name]
        pixels_to_add = np.where(segmentation_image == segment_id)
        if len(pixels_to_add[0]) < 45:
            continue
        section_mask[pixels_to_add] = 1
    section_masks = section_masks

    for key in section_masks.keys():
        try:
            section_mask[key] = skimage.morphology.remove_small_objects(section_mask[key], min_size=64, connectivity=1, in_place=False)
        except:
            pass

    geometries = {}
    for k, v in section_masks.items():
        points = np.array(np.where(v == 1)).T
        if points.shape[0] < 10:
            continue
        convex_hull_data = ConvexHull(points)
        hull_coordinates = points[convex_hull_data.vertices]
        geometries[k] = geometry.Polygon(hull_coordinates)
    return geometries

def make_instances(im1_path, label_file):
    segmentation_file = im1_path.replace(".png", "-segmentation0001.exr")
    data = json.load(open(label_file, "r"))
    sku_name_to_section_name = data["sku_name_to_section_name"]
    index_mapping = data["index_mapping"]
    segment_id_to_section_name = {index_mapping[k]:v for k,v in sku_name_to_section_name.items()}
    segmentation_image = cv2.imread(segmentation_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
    ids_in_image = np.unique(segmentation_image)

    #'''
    section_masks = defaultdict(lambda: np.array(np.zeros((segmentation_image.shape[0], segmentation_image.shape[1]), int)))
    for segment_id in ids_in_image:
        if segment_id not in segment_id_to_section_name:
            continue
        #section_name = segment_id_to_section_name[int(segment_id)]
        section_mask = section_masks[segment_id]
        pixels_to_add = np.where(segmentation_image == segment_id)
        if len(pixels_to_add[0]) < 45:
            continue
        section_mask[pixels_to_add] = 1
    #'''
    '''
    section_masks = defaultdict(lambda: np.array(np.zeros((segmentation_image.shape[0], segmentation_image.shape[1]), int)))
    for segment_id in ids_in_image:
        if segment_id not in segment_id_to_section_name:
            continue
        section_name = segment_id_to_section_name[int(segment_id)]
        section_mask = section_masks[section_name]
        pixels_to_add = np.where(segmentation_image == segment_id)
        if len(pixels_to_add[0]) < 45:
            continue
        section_mask[pixels_to_add] = 1
    section_masks = section_masks
    '''

    geometries = {}
    for k, v in section_masks.items():
        points = np.array(np.where(v == 1)).T
        if points.shape[0] < 10:
            continue
        convex_hull_data = ConvexHull(points)
        hull_coordinates = points[convex_hull_data.vertices]
        geometries[k] = geometry.Polygon(hull_coordinates)
    return geometries

def generate_instance_labels(im1_path, label_file, category_dict):
    segmentation_file = im1_path.replace(".png", "-segmentation0001.exr")
    data = json.load(open(label_file, "r"))
    sku_name_to_section_name = data["sku_name_to_section_name"]
    index_mapping = data["index_mapping"]
    segment_id_to_section_name = {index_mapping[k]:v for k,v in sku_name_to_section_name.items()}
    segmentation_image = cv2.imread(segmentation_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
    ids_in_image = np.unique(segmentation_image)

    bboxes, masks, cat_ids = [], [], []

    section_masks = defaultdict(lambda: np.array(np.zeros((segmentation_image.shape[0], segmentation_image.shape[1]), int)))
    for segment_id in ids_in_image:
        if segment_id not in segment_id_to_section_name:
            continue

        sectionname = segment_id_to_section_name[segment_id]
        corresponding_key = [k for k,v in index_mapping.items() if v == sectionname]
        if len(corresponding_key) == 0:
            continue
        name = [k for k,v in index_mapping.items() if v == sectionname][0].split('.')[0]
        if name not in category_dict.keys():
            continue

        section_mask = section_masks[segment_id]
        pixels_to_add = np.where(segmentation_image == segment_id)
        if len(pixels_to_add[0]) < 45:
            continue
        section_mask[pixels_to_add] = 1
        
        points = np.array(np.where(segmentation_image == segment_id)).T
        if points.shape[0] < 10:
            continue
        # Check if all points have same x or y coordinate
        # For ConvexHull to work must have at least 2
        unique_x = set([p[0] for p in points])
        unique_y = set([p[1] for p in points])
        if len(unique_x) == 1 or len(unique_y) == 1:
            continue
        convex_hull_data = ConvexHull(points)
        hull_coordinates = points[convex_hull_data.vertices]
        masks.append(hull_coordinates.tolist())
        
        i, j = np.where(segmentation_image == segment_id)
        if len(i) < 10:
            continue
        bboxes.append(skimage.measure.regionprops(section_mask)[0].bbox) # (min_row, min_col, max_row, max_col)

        category_id = category_dict[name]
        
        cat_ids.append(category_id)

    return {"name": im1_path, "bbox": bboxes, "masks": masks, "ids": cat_ids}


def generate_category_dict():
    """ Generates dict mapping object names to their ids.
    Note: Only run once.
    """
    with open('synthetic_anno.json', 'r') as jsonFile:
        data = json.load(jsonFile)
    all_cats = list(set([d['name'].split('.')[0] for d in data['categories']]))
    all_cats_dict = dict()
    for i in range(len(all_cats)):
        all_cats_dict[all_cats[i]] = i

    with open('category_ids.json', 'w') as jsonFile:
        json.dump(all_cats_dict, jsonFile)

if __name__ == "__main__":

    with open('category_ids.json', 'r') as jsonFile:
        category_dict = json.load(jsonFile)

    #''' # Make instance seg mask annotations
    data_root = '/app/renders_multicam_diff_1'
    # Generate instance segmentation mask labels
    with open('/app/synthetic_data_baselines/imagesets/val.txt', 'r') as trainFile:
        train_scenes = trainFile.readlines()
    train_scenes = [t[:-1] for t in train_scenes]

    #train_scenes = train_scenes[:1]# TODO: REMOVE AFTER DEBUG
    #for scene in train_scenes:
    for i in tqdm(range(len(train_scenes))):
        scene = train_scenes[i]
    
        # Get image path and label file name
        im_path1 = os.path.join(data_root, scene+'_change-0.png')
        if not Path(f"/app/synthetic_data_baselines/utils/instance_anno/{im_path1.split('/')[-1][:-4]}.json").exists():
            print(im_path1)
            label1 = os.path.join(data_root, scene+'_change-0_label.json')
            # Generate masks
            annotation1 = generate_instance_labels(im_path1, label1, category_dict)
            with open(f"/app/synthetic_data_baselines/utils/instance_anno/{im_path1.split('/')[-1][:-4]}.json", 'w') as jsonFile:
                json.dump(annotation1, jsonFile)
        #if len(np.unique(mask1)) > 255:
        #    print(im_path1, " has more than 255 obj")
        #plt_filename1 = f"/app/synthetic_data_baselines/utils/debug/{im_path1.split('/')[-1]}"
        #img1 = Image.fromarray(mask1).convert('RGB')
        #img1.save(plt_filename1, "png")

        #img1_check = np.array(Image.open(plt_filename1))
        #print("unique ", np.unique(img1_check))

        # Get image path and label file name
        im_path2 = os.path.join(data_root, scene+'_change-1.png')
        if not Path(f"/app/synthetic_data_baselines/utils/instance_anno/{im_path2.split('/')[-1][:-4]}.json").exists():
            print(im_path2)
            label2 = os.path.join(data_root, scene+'_change-1_label.json')
            # Generate masks
            annotation2 = generate_instance_labels(im_path2, label2, category_dict)
            with open(f"/app/synthetic_data_baselines/utils/instance_anno/{im_path2.split('/')[-1][:-4]}.json", 'w') as jsonFile:
                json.dump(annotation2, jsonFile)
        #if len(np.unique(mask2)) > 255:
        #    print(im_path2, " has more than 255 obj")
        #plt_filename2 = f"/app/synthetic_data_baselines/utils/debug/{im_path2.split('/')[-1]}"
        #img2 = Image.fromarray(mask2).convert('RGB')
        #img2.save(plt_filename2, "png")
    #'''

    '''
    # Make camogram annotations
    dir_path = "/app/renders_multicam_diff_1/"
    examples_png = sorted([filename for filename in glob.glob(f'{dir_path}/*.png')])
    examples = []
    for im_path in examples_png:
        label_file = im_path.replace(".png", "_label.json")
        segmentation_file = im_path.replace(".png", "-segmentation0001.exr")
        depth_file = im_path.replace(".png", "-depth0001.exr")
        if os.path.exists(label_file) and os.path.exists(segmentation_file) and os.path.exists(depth_file):
            examples.append((im_path, label_file, segmentation_file, depth_file))

    len(examples)
    
    for example in examples:
        im_path, label_file, segmentation_file, depth_file = example
        img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
        img_black = np.zeros((img.shape[0], img.shape[1], 3))
        geometries = generate_instance_labels(im_path, label_file)
        img = Image.new('RGB', (img.shape[1], img.shape[0]), 0)
        fill_index = 0
        for key, camogram_annotation in geometries.items():
            polygon = list(zip(camogram_annotation.exterior.xy[1], camogram_annotation.exterior.xy[0]))
            ImageDraw.Draw(img, mode="RGB").polygon(polygon, fill=(fill_index // 255, fill_index % 255, 0))
            fill_index += 20
            mask = np.array(img)
            img_black = mask
        print("unique in mask ", np.unique(mask))
        plt_filename = f"/app/synthetic_data_baselines/utils/debug/{im_path.split('/')[-1]}"
        Image.fromarray(mask).save(plt_filename, "png")
    '''




