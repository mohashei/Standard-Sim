import os
import cv2
import time
import json
import numpy as np
from glob import glob
from PIL import Image
from skimage import measure
#from matplotlib.patches import Polygon
from shapely.geometry import Polygon, MultiPolygon
from typing import Any, Dict, List, Optional, Set

from utils.constants import Action


def find_files(root: str) -> List[Dict[str, str]]:
    imgs = sorted([filename for filename in glob(f"{root}/*.png") if "change-0" in filename])
    files = []
    for name in imgs:
        label_file1 = name.replace(".png", "-segmentation0001.exr")
        label_file2 = label_file1.replace("change-0", "change-1")
        label_json1 = name.replace(".png", "_label.json")
        label_json2 = label_json1.replace("change-0", "change-1")
        files_exist = (
            os.path.isfile(label_file1)
            and os.path.isfile(label_file2)
            and os.path.isfile(label_json1)
            and os.path.isfile(label_json2)
        )

        if files_exist:
            bbox_json = name.replace("_change-0.png", "-boxes.json")
            label_file = name.replace("_change-0.png", "-label.png")
            files.append(
                {
                    "label1": label_file1,
                    "label2": label_file2,
                    "label": label_file,
                    "label1_json": label_json1,
                    "label2_json": label_json2,
                    "bbox_json": bbox_json,
                }
            )
    return files

def make_bbox_camogram(
    label_file1: str, label_file2: str, segmentation_file1: str, segmentation_file2: str
) -> Dict[str, List[List[int]]]:
    data1 = json.load(open(label_file1, "r"))
    data2 = json.load(open(label_file2, "r"))
    sku_name_to_section_name1 = data1["sku_name_to_section_name"]
    sku_name_to_section_name2 = data2["sku_name_to_section_name"]

    index_mapping1 = data1["index_mapping"]
    shifted_skus1 = {index_mapping1[k] for k in data1["shifted_skus"]}
    shifted_skus_small1 = {index_mapping1[k] for k in data1["shifted_skus_small"]}

    index_mapping2 = data2["index_mapping"]
    added_skus2 = {index_mapping2[k] for k in data2["added_skus"]}
    removed_skus2 = {index_mapping2[k] for k in data2["removed_skus"]}
    shifted_skus2 = {index_mapping2[k] for k in data2["shifted_skus"]}
    shifted_skus_small2 = {index_mapping2[k] for k in data2["shifted_skus_small"]}

    added_skus = added_skus2
    removed_skus = removed_skus2
    shifted_skus = shifted_skus1.union(shifted_skus2)
    shifted_skus_small = shifted_skus_small1.union(shifted_skus_small2)

    segment_id_to_section_name1 = {index_mapping1[k] for k in sku_name_to_section_name1.keys()}
    segment_id_to_section_name2 = {index_mapping2[k] for k in sku_name_to_section_name2.keys()}
    segment_id_to_section_name = segment_id_to_section_name1.union(segment_id_to_section_name2)

    segmentation_image1 = cv2.imread(segmentation_file1, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[
        :, :, 0
    ]
    segmentation_image2 = cv2.imread(segmentation_file2, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[
        :, :, 0
    ]
    H, W = segmentation_image1.shape[:2]
    ids_in_image1 = np.unique(segmentation_image1).astype(int)
    ids_in_image2 = np.unique(segmentation_image2).astype(int)
    ids_in_image = np.unique(np.concatenate((ids_in_image1, ids_in_image2), axis=0))
    change_boxes_json = []
    small_change_boxes_json = []
    no_change_boxes_json = []
    for segment_id in ids_in_image:
        if segment_id not in segment_id_to_section_name:
            continue
        if segment_id in added_skus:
            action = Action.ADDED.value
            segmentation_image = segmentation_image2
        elif segment_id in removed_skus:
            action = Action.REMOVED.value
            segmentation_image = segmentation_image1
        elif segment_id in shifted_skus:
            action = Action.SHIFTED.value
            segmentation_image = segmentation_image2
        elif segment_id in shifted_skus_small:
            action = Action.SHIFTED_SMALL.value
            segmentation_image = segmentation_image2
        else:
            action = Action.NULL.value
            segmentation_image = segmentation_image2
        pixels_to_add = np.where(segmentation_image == segment_id)
        if len(pixels_to_add[0]) < 45:
            continue
        tlbr = np.array(
            [
                np.min(pixels_to_add[0]),
                np.min(pixels_to_add[1]),
                np.max(pixels_to_add[0]),
                np.max(pixels_to_add[1]),
            ]
        )
        area = (tlbr[2] - tlbr[0]) * (tlbr[3] - tlbr[1])
        if area > 64:
            normed_box = [tlbr[0] / H, tlbr[1] / W, tlbr[2] / H, tlbr[3] / W, action]
            if action in [Action.ADDED.value, Action.REMOVED.value, Action.SHIFTED.value]:
                change_boxes_json.append(normed_box)
            elif segment_id == Action.SHIFTED_SMALL.value:
                small_change_boxes_json.append(normed_box)
            else:
                no_change_boxes_json.append(normed_box)
    boxes_json = {
        "change_boxes": change_boxes_json,
        "small_change_boxes": small_change_boxes_json,
        "no_change_boxes": no_change_boxes_json,
    }
    return boxes_json

def make_masks(
    change_mask,
    item_set: Set[str],
    label_json: Dict[str, Any],
    label,
    action: str,
    label2_json: Optional[Dict[str, Any]] = None,
    label2 = None,
):
    if action == "take" or action == "put":
        for item in item_set:
            index = label_json["index_mapping"][item]
            mask = label == index
            change_mask = np.logical_or(change_mask, mask)

    elif action == "shift":
        assert label2_json is not None and label2 is not None
        for item in item_set:
            index1 = label_json["index_mapping"][item]
            mask1 = label == index1
            index2 = label2_json["index_mapping"][item]
            mask2 = label2 == index2
            # Comment next three lines to get union of before/after for shifts
            mask = np.logical_and(mask2, np.logical_not(mask1)) 
            change_mask = np.logical_or(change_mask, mask)
            change_mask = np.logical_and(mask1, mask2) # Use union of masks for shift
    return change_mask

def make_polygon_list(
    change_mask,
    item_set: Set[str],
    label_json: Dict[str, Any],
    label,
    action: str,
    label2_json: Optional[Dict[str, Any]] = None,
    label2 = None,
):
    poly_list = []
    if action == "take" or action == "put":
        for item in item_set:
            index = label_json["index_mapping"][item]
            mask = label == index
            if np.any(mask):
                #poly_list.append(Polygon(np.array([*zip(*np.where(mask == True))])))
                poly_list.append(np.array([*zip(*np.where(mask == True))]).tolist())
    elif action == "shift":
        assert label2_json is not None and label2 is not None
        for item in item_set:
            index1 = label_json["index_mapping"][item]
            mask1 = label == index1
            index2 = label2_json["index_mapping"][item]
            mask2 = label2 == index2
            # Comment next two lines to get union of before/after for shifts
            mask = np.logical_and(mask2, np.logical_not(mask1))
            if np.any(mask):
                #poly_list.append(Polygon(np.array([*zip(*np.where(mask == True))])))
                poly_list.append(np.array([*zip(*np.where(mask == True))]).tolist())
    return poly_list

def create_dict(data_root):
    """ Create dataset dictionary with the following fields for each sample:
    "name": <store>_<id>_cam-<cam_num> (str)
    "height": height (int)
    "width": width (int)
    "bbox": [x, y, h, w] (List[int, int, int, int])
    "take": Polygon
    "put": Polygon
    "shift": Polygon
    TODO
    "SKU_ID": alphanumeric string
    "depth_before": Numpy array shape (h,w) 
    "depth_after": Numpy array shape (h,w)
    "camera_intrinsics"
    "camera_extrinsics"
    """
    all_samples = dict()
    all_samples["data"] = []

    files = find_files(data_root)
    for file in files[:5]:
        dataset_dict = dict()
        # Add scene name
        dataset_dict["name"] = '_'.join(file['label1'].split('/')[-1].split('_')[:3])
        print(dataset_dict["name"])

        # Read relevant files and extract item info
        label1 = cv2.imread(file['label1'], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
        label2 = cv2.imread(file['label2'], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
        label1_json = file["label1_json"]
        with open(label1_json) as f:
            label1_json = json.load(f)

        label2_json = file["label2_json"]
        with open(label2_json) as f:
            label2_json = json.load(f)

        items1_removed = set(label1_json["removed_skus"])
        items1_added = set(label1_json["added_skus"])
        items2_removed = set(label2_json["removed_skus"])
        items2_added = set(label2_json["added_skus"])
        items2_shifted = set(label2_json["shifted_skus"])

        # Add height and width
        height, width = label1.shape
        dataset_dict["height"] = height
        dataset_dict["width"] = width

        # Changed item bounding boxes
        all_boxes = make_bbox_camogram(
            file["label1_json"], file["label2_json"], file["label1"], file["label2"]
        )
        dataset_dict["bbox"] = all_boxes["change_boxes"]
        dataset_dict["bbox"].extend(all_boxes["small_change_boxes"])

        # Change masks
        change_mask_added = np.zeros_like(label1)
        change_mask_removed = np.zeros_like(label1)
        change_mask_shifted = np.zeros_like(label1)

        #change_mask_added = make_masks(change_mask_added, items1_removed, label2_json, label2, "put")
        #change_mask_removed = make_masks(change_mask_removed, items1_added, label1_json, label1, "take")
        #change_mask_removed = make_masks(change_mask_removed, items2_removed, label1_json, label1, "take")
        
        #change_mask_added = make_masks(change_mask_added, items2_added, label2_json, label2, "put")
        #change_mask_shifted = make_masks(
        #    change_mask_shifted,
        #    items2_shifted,
        #    label1_json,
        #    label1,
        #    "shift",
        #    label2_json=label2_json,
        #    label2=label2,
        #)

        change_mask = np.dstack((change_mask_removed, change_mask_added, change_mask_shifted))
        dataset_dict["put"] = make_polygon_list(change_mask_added, items1_removed, label2_json, label2, "put")
        dataset_dict["take"] = make_polygon_list(change_mask_removed, items1_added, label1_json, label1, "take")
        dataset_dict["take"].extend(make_polygon_list(change_mask_removed, items2_removed, label1_json, label1, "take"))
        dataset_dict["put"].extend(make_polygon_list(change_mask_added, items2_added, label2_json, label2, "put"))
        dataset_dict["shift"] = make_polygon_list(change_mask_shifted, items2_shifted, label1_json, label1, "shift", label2_json, label2)

        all_samples["data"].append(dataset_dict)
    return all_samples

def visualize_masks(anno_sample):
    """ Save mask given points from anno file
    """

    mask = np.zeros((anno_sample['height'], anno_sample['width']))
    takes = anno_sample["take"] # List of Nx2 lists
    for t in takes:
        t = np.array(t).transpose()
        flat_t = np.ravel_multi_index(t, mask.shape)
        np.ravel(mask)[flat_t] = 1
    puts = anno_sample["put"]
    for p in puts:
        p = np.array(p).transpose()
        flat_p = np.ravel_multi_index(p, mask.shape)
        np.ravel(mask)[flat_p] = 2
    shifts = anno_sample["shift"]
    for s in shifts:
        s = np.array(s).transpose()
        flat_s = np.ravel_multi_index(s, mask.shape)
        np.ravel(mask)[flat_s] = 3

    print(os.path.join('/app/synthetic_data_baselines/utils/debug/', anno_sample['name']+'.png'))
    #mask_color = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    #cv2.imwrite(os.path.join('/app/synthetic_data_baselines/utils/debug/', anno_sample['name']+'.png'), mask_color)
    
    ''' # Using PIL Image
    # Get color palette to visualize labels
    img = Image.open("/app/renders_multicam_diff_1/circ.us.0000_8b7831e9036a424d91c83bd9e4de50e0_cam-2_change-1.png").convert('P', palette=Image.ADAPTIVE, colors=255)
    color_palette = img.getpalette()
    # Set background color to black
    color_palette[:3] = [0, 0, 0]
    mask = Image.fromarray(mask)
    mask.putpalette(color_palette)
    mask.save(os.path.join('/app/synthetic_data_baselines/utils/debug/', anno_sample['name']+'.png'))
    '''

def create_sub_masks(mask_image):
    """
    @param mask_image (PIL Image)
    """
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_submask_from_array(mask_image):
    """
    @param mask_image (Numpy array)
    """
    width, height = mask_image.shape

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image[x][y]

            # If the pixel is not black...
            if pixel != 0:
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd, scene):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        #segmentation = np.array(poly.exterior.coords).ravel().tolist()
        if poly.type=='MultiPolygon':
            segmentation = [list(x.exterior.coords) for x in poly.geoms]
        else:
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
        if len(segmentation) > 0:
            polygons.append(poly)
            segmentations.append(segmentation)

    new_poly = []
    for poly in polygons:
        if poly.type=='MultiPolygon':
            new_poly.extend(list(poly))
        else:
            new_poly.append(poly)
    polygons = new_poly
    if len(polygons) > 0:
        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        #y, x, max_y, max_x = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area

        annotation = {
            'segmentation': segmentations,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': bbox,
            'area': area,
            'scene': scene
        }

        return annotation
    else:
        return None

if __name__ == "__main__":
    data_root = "/renders_multicam_diff_1"

    anno_dict = dict({"images": [], "annotations": [], "categories": []})

    files = find_files(data_root)
    #files = [{'label1': '/app/renders_multicam_diff_1/circ.us.0000_0017c0b964c2492db349f0591c6af20a_cam-1_change-0-segmentation0001.exr', 'label2': '/app/renders_multicam_diff_1/circ.us.0000_0017c0b964c2492db349f0591c6af20a_cam-1_change-1-segmentation0001.exr', 'label': '/app/renders_multicam_diff_1/circ.us.0000_0017c0b964c2492db349f0591c6af20a_cam-1-label.png', 'label1_json': '/app/renders_multicam_diff_1/circ.us.0000_0017c0b964c2492db349f0591c6af20a_cam-1_change-0_label.json', 'label2_json': '/app/renders_multicam_diff_1/circ.us.0000_0017c0b964c2492db349f0591c6af20a_cam-1_change-1_label.json', 'bbox_json': '/app/renders_multicam_diff_1/circ.us.0000_0017c0b964c2492db349f0591c6af20a_cam-1-boxes.json'}]
    #files = [{'label1': '/app/renders_multicam_diff_1/cmps.ca.0000_f7fb7fe8a30e41b9b6ec838bb4135e65_cam-1_change-0-segmentation0001.exr', 'label2': '/app/renders_multicam_diff_1/cmps.ca.0000_f7fb7fe8a30e41b9b6ec838bb4135e65_cam-1_change-1-segmentation0001.exr', 'label': '/app/renders_multicam_diff_1/cmps.ca.0000_f7fb7fe8a30e41b9b6ec838bb4135e65_cam-1-label.png', 'label1_json': '/app/renders_multicam_diff_1/cmps.ca.0000_f7fb7fe8a30e41b9b6ec838bb4135e65_cam-1_change-0_label.json', 'label2_json': '/app/renders_multicam_diff_1/cmps.ca.0000_f7fb7fe8a30e41b9b6ec838bb4135e65_cam-1_change-1_label.json', 'bbox_json': '/app/renders_multicam_diff_1/cmps.ca.0000_f7fb7fe8a30e41b9b6ec838bb4135e65_cam-1-boxes.json'}]

    # i is index into images, j index into annotations, k index into categories
    i, j, k = 0, 0, 0
    skus = set()
    for file in files:
        print(file)
        scene_name='_'.join(file['label1'].split('/')[-1].split('_')[:3])
        # (From preprocess_synthetic_data.py)
        label_file1 = file["label1"]
        label_file2 = file["label2"]
        label1 = cv2.imread(label_file1, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
        label2 = cv2.imread(label_file2, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
        label1_json = file["label1_json"]
        with open(label1_json) as f:
            label1_json = json.load(f)
        label2_json = file["label2_json"]
        with open(label2_json) as f:
            label2_json = json.load(f)
        items1_removed = set(label1_json["removed_skus"]) # Should be empty
        items1_added = set(label1_json["added_skus"]) # Should be empty
        items2_removed = set(label2_json["removed_skus"])
        items2_added = set(label2_json["added_skus"])
        items2_shifted = set(label2_json["shifted_skus"])
        bounding_boxes = make_bbox_camogram(
            file["label1_json"], file["label2_json"], file["label1"], file["label2"]
        )
        change_mask_added = np.zeros_like(label1)
        change_mask_removed = np.zeros_like(label1)
        change_mask_shifted = np.zeros_like(label1)
        change_mask_added = make_masks(change_mask_added, items1_removed, label2_json, label2, "put")
        change_mask_removed = make_masks(change_mask_removed, items1_added, label1_json, label1, "take")
        change_mask_removed = make_masks(
            change_mask_removed, items2_removed, label1_json, label1, "take"
        )
        change_mask_added = make_masks(change_mask_added, items2_added, label2_json, label2, "put")
        change_mask_shifted = make_masks(
            change_mask_shifted,
            items2_shifted,
            label1_json,
            label1,
            "shift",
            label2_json=label2_json,
            label2=label2,
        )
        change_mask = np.dstack((change_mask_removed, change_mask_added, change_mask_shifted))

        # Create dict entry in images list
        height, width = label1.shape
        image1_name = scene_name+"_change-0.png"
        image2_name = scene_name+"_change-1.png"
        randommat1_name = scene_name+"_change-0-randommats.png"
        randommat2_name = scene_name+"_change-1-randommats.png"
        depth1_name = scene_name+"_change-0-depth0001.exr"
        depth2_name = scene_name+"_change-1-depth0001.exr"
        image_dict = dict({
            "id": i, 
            "license": 1,
            "scene": scene_name,
            "width": width, 
            "height": height, 
            "randommats1": randommat1_name,
            "randommats2": randommat2_name,
            "depth1": depth1_name, 
            "depth2": depth2_name, 
            "image1": image1_name, 
            "image2": image2_name})
        anno_dict["images"].append(image_dict)

        # Make a mask for every added SKU
        for item in items2_added:
            change_mask_added = np.zeros_like(label1)
            change_mask_added = make_masks(change_mask_added, set({item}), label2_json, label2, "put")
            if np.any(change_mask_added):
                submasks = create_submask_from_array(change_mask_added)
                for color, sub_mask in submasks.items():
                    if item not in skus:
                        skus.add(item)
                        anno_dict["categories"].append({"id": k, "name": item})
                        sku_id = k
                        k+=1
                    else:
                        sku_id = [d.get("id") for d in anno_dict["categories"] if d["name"] == item][0]
                    anno = create_sub_mask_annotation(np.array(sub_mask), image_id=i, category_id=sku_id, annotation_id=j, is_crowd=0, scene=scene_name)
                    if anno:
                        anno["action"] = "put"
                        anno_dict["annotations"].append(anno)
                        print("put ", sku_id)
                        j+=1
        # Make a mask for every removed SKU
        for item in items2_removed:
            change_mask_removed = np.zeros_like(label1)
            change_mask_removed = make_masks(change_mask_removed, set({item}), label2_json, label2, "put")
            if np.any(change_mask_removed):
                submasks = create_submask_from_array(change_mask_removed)
                for color, sub_mask in submasks.items():
                    if item not in skus:
                        skus.add(item)
                        anno_dict["categories"].append({"id": k, "name": item})
                        sku_id = k
                        k+=1
                    else:
                        sku_id = [d.get("id") for d in anno_dict["categories"] if d["name"] == item][0]
                    anno = create_sub_mask_annotation(np.array(sub_mask), image_id=i, category_id=sku_id, annotation_id=j, is_crowd=0, scene=scene_name)
                    if anno:
                        anno["action"] = "take"
                        anno_dict["annotations"].append(anno)
                        print("take ", sku_id)
                        j+=1
        # Make a mask for every shifted SKU
        for item in items2_shifted:
            change_mask_shifted = np.zeros_like(label1)
            change_mask_shifted = make_masks(change_mask_shifted, set({item}), label2_json, label2, "put")
            if np.any(change_mask_shifted):
                submasks = create_submask_from_array(change_mask_shifted)
                for color, sub_mask in submasks.items():
                    if item not in skus:
                        skus.add(item)
                        anno_dict["categories"].append({"id": k, "name": item})
                        sku_id = k
                        k+=1
                    else:
                        sku_id = [d.get("id") for d in anno_dict["categories"] if d["name"] == item][0]
                    anno = create_sub_mask_annotation(np.array(sub_mask), image_id=i, category_id=sku_id, annotation_id=j, is_crowd=0, scene=scene_name)
                    if anno:
                        anno["action"] = "shift"
                        anno_dict["annotations"].append(anno)
                        print("shift ", sku_id)
                        j+=1
        i+=1


    # Add info
    anno_dict["info"] = dict({
        "description": "Synthetic dataset created in Blender for change detection, instance segmentation, and depth estimation.",
        "url": "https://github.com/Standard-Cognition/blender-synth",
        "version": 1,
        "year": 2021,
        "contributor": "Cristina Mata, Nick Locascio, Mohammed Sheikh, Kenneth Kihara",
        "date_created": "July 1, 2021"
    })
    # Add license
    anno_dict["licenses"] = [dict({
        "url": "url_to_our_license",
        "id": 1,
        "name": "Attribution License"
    })]

    with open("/app/synthetic_data_baselines/utils/synthetic_anno.json", 'w') as jsonFile:
        json.dump(anno_dict, jsonFile)

