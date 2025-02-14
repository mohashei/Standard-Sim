import os
import re
import cv2
import glob
import json
import time
import random
import string
import imageio
import collections
import numpy as np
from PIL import Image
from typing import Tuple
from shapely.geometry import Polygon

import torch

from synthetic_data_baselines.datasets.base_dataset import BaseDataset
from synthetic_data_baselines.utils.synthetic_dataset_tools import SyntheticDataset
from synthetic_data_baselines.datasets.utils import convert_action_to_int
from datasets.augment import (
    add_noise,
    box_augment,
    flip_lr,
    flip_pair,
    mask2bbox,
)

class SyntheticPairsDataSet(BaseDataset):
    """Dataset of synthetic image pairs and change masks from synthetic images generated by blender
    Using official splits from imagesets dir.
    """

    DELIM_CHAR = ["-", "_"]
    IMG_FILE1 = "0.png"  # _1.png
    IMG_FILE2 = "1.png"  # _2.png
    LABEL_FILE = "_label.png"
    BBOX_FILE = "_bbox.json"

    def __init__(self, configuration):
        self.root = configuration["root"]
        self.split = configuration["mode"]
        assert self.split in ["train", "val", "test"]
        self.files = collections.defaultdict(list)
        self.img_normalize = configuration["normalize"]
        self.crops = configuration["crop"]
        self.resize = configuration["resize"]
        self.spatial_resolution = configuration["spatial_resolution"]

        if not os.path.exists('/root/.imageio/freeimage'):
            imageio.plugins.freeimage.download()

        # For overfitting always use first 4 samples
        if configuration["overfit"]:
            print("Overfitting to 4 samples")
            with open(os.path.join('/app/synthetic_data_baselines/imagesets/train.txt'), 'r') as imgsetFile:
                scenes = imgsetFile.readlines()
            scenes = [e[:-1]for e in scenes]
            scenes.sort()
            scenes = scenes[:4]
        else:
            with open(os.path.join('/app/synthetic_data_baselines/imagesets', self.split+'.txt'), 'r') as imgsetFile:
                scenes = imgsetFile.readlines()
            scenes = [e[:-1]for e in scenes]

        anno_data = SyntheticDataset('/app/synthetic_data_baselines/utils/synthetic_anno.json', '/app/renders_multicam_diff_1')
        image_ids = anno_data.getImgIds(scenes)
        ann_ids = anno_data.getAnnIds(image_ids)

        info = anno_data.mapScenesToInfo(scenes)
        for k,v in info.items():
            self.files[self.split].append(
                {
                    "img1": os.path.join(self.root, v["img1"]),
                    "img2": os.path.join(self.root, v["img2"]),
                    "label": v["label"],
                    "actions": convert_action_to_int(v["actions"]),
                    "bbox": v["bbox"],
                    "depth1": os.path.join(self.root, v["depth1"]),
                    "depth2": os.path.join(self.root, v["depth2"])
                }
            )

        self.mean = torch.tensor([0.406 * 255, 0.456 * 255, 0.485 * 255]).view(3, 1, 1)
        self.std = torch.tensor([0.225 * 255, 0.224 * 255, 0.229 * 255]).view(3, 1, 1)
        self.mean_depth = torch.tensor(7.7304)
        self.depth_std = torch.tensor(1.4214)

        self.res_x = self.spatial_resolution[0]
        self.res_y = self.spatial_resolution[1]

        self.aug = configuration["augment"]

        self.augment_params = {
            "box_augment_params": {
                "max_boxes": 3,
                "min_height_mult": 0.1,
                "max_height_mult": 0.5,
                "min_width_mult": 0.1,
                "max_width_mult": 0.3,
                "sat_prob": 0.5,
                "sat_min": 0.5,
                "sat_max": 1.5,
                "brightness_prob": 0.5,
                "brightness_min": 0.5,
                "brightness_max": 1.5,
            },
            "noise_params": {
                "qe_low": 0.65,
                "qe_high": 0.72,
                "bit_depth": 8,
                "baseline": 0,
                "sensitivity_low": 1.2,
                "sensitivity_high": 1.7,
                "dark_noise_low": 2.5,
                "dark_noise_high": 3.5,
            },
            "flip_pair": 0.0,
            "flip_lr": 0.5,
        }


    def __len__(self):
        return len(self.files[self.split])

    def augment(
        self, img1, img2, label, bboxes, depth1, depth2
    ):
        img1, img2 = add_noise(img1, img2, self.augment_params["noise_params"])
        img1, img2 = box_augment(img1, img2, self.augment_params["box_augment_params"])
        if random.random() < self.augment_params["flip_pair"]:
            img1, img2, bboxes, depth1, depth2 = flip_pair(img1, img2, bboxes, depth1, depth2)
        if random.random() < self.augment_params["flip_lr"]:
            img1, img2, label, bboxes, depth1, depth2 = flip_lr(img1, img2, label, bboxes, depth1, depth2)
        return img1, img2, label, bboxes, depth1, depth2

    def resize_imgs(
        self, img1, img2, label, depth1, depth2
    ):
        img1 = cv2.resize(img1, (self.res_x, self.res_y))
        img2 = cv2.resize(img2, (self.res_x, self.res_y))
        depth1 = cv2.resize(depth1, (self.res_x, self.res_y))
        depth2 = cv2.resize(depth2, (self.res_x, self.res_y))
        label = cv2.resize(label, (self.res_x, self.res_y), interpolation=cv2.INTER_NEAREST)
        return img1, img2, label, depth1, depth2

    def crop(
        self, img1, img2, label, bboxes, depth1, depth2
    ):
        height, width = img1.shape[:2]

        bbox = mask2bbox(label)
        if bbox is not None:
            change_rmin, change_cmin, change_rmax, change_cmax = bbox
        else:
            change_rmin, change_cmin, change_rmax, change_cmax = 0, 0, height, width

        bbheight = change_rmax - change_rmin
        bbwidth = change_cmax - change_cmin

        if bbheight < self.res_y:
            rmin = max(change_rmin - self.res_y + bbheight, 0)
            rmax = change_rmin
        else:
            rmin = change_rmin
            rmax = change_rmin - self.res_y + bbheight
        if bbwidth < self.res_x:
            cmin = max(change_cmin - self.res_x + bbwidth, 0)
            cmax = change_cmin
        else:
            cmin = change_cmin
            cmax = change_cmin - self.res_x + bbwidth

        x = random.randint(cmin, cmax)
        y = random.randint(rmin, rmax)
        x = min(x, width - self.res_x - 1)
        y = min(y, height - self.res_y - 1)
        ymax = y + self.res_y
        xmax = x + self.res_x
        img1 = img1[y:ymax, x:xmax, :]
        img2 = img2[y:ymax, x:xmax, :]
        label = label[y:ymax, x:xmax]
        depth1 = depth1[y:ymax, x:xmax]
        depth2 = depth2[y:ymax, x:xmax]
        # TODO: Modify bbox cropping to account for point order
        '''
        bboxes[:, :4] *= np.array([height, width, height, width])
        ya = np.fmax(bboxes[:, 0], y)
        xa = np.fmax(bboxes[:, 1], x)
        yb = np.fmin(bboxes[:, 2], ymax)
        xb = np.fmin(bboxes[:, 3], xmax)

        inds = np.logical_and((xb - xa) > 0, (yb - ya) > 0)
        xa, xb, ya, yb = xa[inds], xb[inds], ya[inds], yb[inds]
        bboxes = bboxes[inds, :]
        bboxes[:, 0] = ya - y
        bboxes[:, 1] = xa - x
        bboxes[:, 2] = yb - y
        bboxes[:, 3] = xb - x
        bboxes[:, :4] /= np.array([self.res_y, self.res_x, self.res_y, self.res_x])
        '''
        return img1, img2, label, bboxes, depth1, depth2

    def get_store(self, index: int):
        store = os.path.basename(self.files[self.split][index]["img1"]).split("_")[0]
        return store

    def __getitem__(self, index: int):
        datafiles = self.files[self.split][index]

        # Images
        img_file1, img_file2 = datafiles["img1"], datafiles["img2"]
        
        img1, img2 = (
            cv2.imread(img_file1),
            cv2.imread(img_file2),
        )
        height, width, _ = img1.shape
        
        # Label
        label_shifts, label_takes, label_puts = np.zeros((height, width)), np.zeros((height, width)), np.zeros((height, width))
        for i in range(len(datafiles["label"])):
            segmentation_points = datafiles["label"][i]
            action = datafiles["actions"][i]
            if action == 1:
                for contour in segmentation_points:
                    # Switching x,y coords in segmentation points
                    formatted_points = []
                    for j in range(0, len(contour), 2):
                        formatted_points.append((contour[j+1], contour[j]))
                    polygon = Polygon(formatted_points)
                    int_coords = lambda x: np.array(x).round().astype(np.int32)
                    exterior = [int_coords(polygon.exterior.coords)]
                    label_takes = cv2.fillPoly(label_takes, exterior, action)
            elif action == 2:
                for contour in segmentation_points:
                    # Switching x,y coords in segmentation points
                    formatted_points = []
                    for j in range(0, len(contour), 2):
                        formatted_points.append((contour[j+1], contour[j]))
                    polygon = Polygon(formatted_points)
                    int_coords = lambda x: np.array(x).round().astype(np.int32)
                    exterior = [int_coords(polygon.exterior.coords)]
                    label_puts = cv2.fillPoly(label_puts, exterior, action)
            elif action == 3:
                for contour in segmentation_points:
                    # Switching x,y coords in segmentation points
                    formatted_points = []
                    for j in range(0, len(contour), 2):
                        formatted_points.append((contour[j+1], contour[j]))
                    polygon = Polygon(formatted_points)
                    int_coords = lambda x: np.array(x).round().astype(np.int32)
                    exterior = [int_coords(polygon.exterior.coords)]
                    label_shifts = cv2.fillPoly(label_shifts, exterior, action)
            else:
                pass
        # Give priority to takes, then puts, then shifts
        label = np.where(label_takes != 0, label_takes, label_puts)
        label = np.where(label != 0, label, label_shifts)

        # TODO: Bounding Boxes
        bboxes = datafiles["bbox"]
        
        # Depth
        depth1, depth2 = (
            imageio.imread(datafiles["depth1"], format="EXR-FI"),
            imageio.imread(datafiles["depth2"], format="EXR-FI")
        )

        if self.crops and not self.resize:
            img1, img2, label, bboxes, depth1, depth2 = self.crop(img1, img2, label, bboxes, depth1, depth2)
        elif self.resize and not self.crops:
            img1, img2, label, depth1, depth2 = self.resize_imgs(img1, img2, label, depth1, depth2)
        elif self.crops and self.resize:
            if random.random() > 0.5:
                img1, img2, label, depth1, depth2 = self.resize_imgs(img1, img2, label, depth1, depth2)
            else:
                img1, img2, label, bboxes, depth1, depth2 = self.crop(img1, img2, label, bboxes, depth1, depth2)
        if self.aug:
            img1, img2, label, bboxes, depth1, depth2 = self.augment(img1, img2, label, bboxes, depth1, depth2)

        # Debug: save images and labels
        scene = '_'.join(datafiles["img1"].split('/')[-1].split('_')[:3])
        cv2.imwrite('debug/{}_img1.jpg'.format(scene), img1)
        cv2.imwrite('debug/{}_img2.jpg'.format(scene), img2)
        cv2.imwrite('debug/{}_label.jpg'.format(scene), label)
        #imageio.show_formats()
        #imageio.imwrite('debug/{}_depth1.exr'.format(tmp_name), depth1.astype("float32"), format="EXR-FI")
        #imageio.imwrite('debug/{}_depth2.exr'.format(tmp_name), depth2.astype("float32"), format="EXR-FI")
        
        img1 = torch.from_numpy(img1).permute(2, 0, 1).contiguous()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).contiguous()
        label = torch.from_numpy(label)
        depth1 = torch.from_numpy(depth1).contiguous()
        depth2 = torch.from_numpy(depth2).contiguous()
        depth1 = torch.unsqueeze(depth1, dim=0)
        depth2 = torch.unsqueeze(depth2, dim=0)

        '''
        bboxes[:, 0] = (bboxes[:, 2] + bboxes[:, 0]) / 2
        bboxes[:, 1] = (bboxes[:, 3] + bboxes[:, 1]) / 2
        bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) * 2
        bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) * 2
        bboxes = bboxes[0:4, :4]
        bboxes[:, 0:2] -= 0.5
        '''
        bboxes = torch.from_numpy(np.array(bboxes))
        

        if self.img_normalize:
            img1 = (img1 - self.mean) / self.std
            img2 = (img2 - self.mean) / self.std
            depth1 = (depth1 - self.mean_depth) / self.depth_std
            depth2 = (depth2 - self.mean_depth) / self.depth_std

        return img1, img2, label, '_'.join(datafiles["img1"].split('/')[-1].split('_')[:3]), depth1, depth2

    @staticmethod
    def collate_fn(batch):
        img1, img2, label, scene, depth1, depth2 = zip(*batch)
        return (
            torch.stack(img1, 0),
            torch.stack(img2, 0),
            torch.stack(label, 0),
            #torch.nn.utils.rnn.pad_sequence(bboxes, batch_first=True, padding_value=-1.0),
            scene,
            torch.stack(depth1, 0),
            torch.stack(depth2, 0)
        )




