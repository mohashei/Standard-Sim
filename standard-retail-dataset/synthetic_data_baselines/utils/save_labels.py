import cv2
import imageio
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

from utils.constants import Action

def write_depth(path, depth, bits=1):
    """Write depth map to pfm and png file.
    Modified from https://github.com/isl-org/MiDaS/blob/master/utils.py
    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))
    else:
        print("Number of bits is not recognized.")



def save_change_det_labels(data_root, label_save):
    """ Saves labels for change detection in color for better visualization.
    @param data_root (str): Path to the root dataset directory, e.g. "/app/renders_multicam_diff_1"
    @param label_save (str): Path to the directory where to save labels, e.g. "/app/change_labels_color"
    """
    palette = [
            0, 0, 0,
            159, 4, 22,
            98, 190, 48,
            122, 130, 188
        ]

    lb_files = [join(data_root, f) for f in listdir(data_root) if isfile(join(data_root, f)) and 'label' in f and 'change' not in f]
    for lb in lb_files:
        label = cv2.imread(lb)
        removed = label[:, :, 0]
        added = np.where(label[:, :, 1], Action.ADDED.value, 0)
        shifted = np.where(label[:, :, 2], Action.SHIFTED.value, 0)
        # added gets priority (we need to break the tie), then removed, then shifts
        label = np.where(added, added, np.where(removed, removed, shifted)).astype(np.uint8)

        
        img1 = Image.fromarray(label.astype(np.uint8))
        img1.putpalette(palette * 64)
        img1.save(join(label_save, lb.split('/')[-1][:-4]+"_color.png"))
        print("Saving ", join(label_save, lb.split('/')[-1][:-4]+"_color.png"))

if __name__ == "__main__":
    data_root = "/app/renders_multicam_diff_1"
    depth_save = "/app/depth_images"

    depth_files = [join(data_root, f) for f in listdir(data_root) if isfile(join(data_root, f)) and 'depth' in f]

    for depth in depth_files:
        depth1 = imageio.imread(depth, format="EXR-FI")

        # Replace 'infinite' depth with max
        depth1[depth1>100] = -1
        depth1[depth1==-1] = np.max(depth1)

        depth_path = join(depth_save, depth.split('/')[-1][:-4])
        write_depth(depth_path, depth1)






