import random
from typing import Dict, Union

import cv2
import numpy as np


def _convert_GRGB_to_RGB(data):
    data = np.uint8(data)

    # Use OpenCV to convert Bayer GRGB to RGB
    data = cv2.cvtColor(data, cv2.COLOR_BayerRG2BGR)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = downsample_subsample(data)
    return data


def to_bayer(srcArray):
    # Open image and put it in a numpy array
    w, h, _ = srcArray.shape
    # Create target array, twice the size of the original image
    resArray = np.zeros((2 * w, 2 * h, 3), dtype=np.uint8)
    # Map the RGB values in the original picture according to the BGGR pattern#
    # Blue
    resArray[::2, ::2, 2] = srcArray[:, :, 2]
    # Green (top row of the Bayer matrix)
    resArray[1::2, ::2, 1] = srcArray[:, :, 1]
    # Green (bottom row of the Bayer matrix)
    resArray[::2, 1::2, 1] = srcArray[:, :, 1]
    # Red
    resArray[1::2, 1::2, 0] = srcArray[:, :, 0]
    return resArray


def downsample_subsample(data):
    return data[::2, ::2, ...]


# Function to add camera noise
def add_camera_noise(
    input_irrad_photons,
    qe: float = 0.69,
    sensitivity: float = 5.88,
    dark_noise: float = 2.29,
    bitdepth: int = 12,
    baseline: int = 100,
    rs: np.random.RandomState = np.random.RandomState(seed=42),
):
    # Add shot noise
    photons = rs.poisson(input_irrad_photons, size=input_irrad_photons.shape)

    # Convert to electrons
    electrons = qe * photons

    # Add dark noise
    electrons_out = rs.normal(scale=dark_noise, size=electrons.shape) + electrons

    # Convert to ADU and add baseline
    max_adu = np.int(2 ** bitdepth - 1)
    adu = (electrons_out * sensitivity).astype(np.int)  # Convert to discrete numbers
    adu += baseline
    adu[adu > max_adu] = max_adu  # models pixel saturation

    return np.clip(adu, 0, 255)


def box_augment(
    img1, img2, augment_params: Dict[str, Union[int, float]]
):
    height, width = img1.shape[:2]
    nbox = augment_params["max_boxes"]
    box_hw = []
    for i in range(2 * nbox):
        box_hw.append(
            (
                random.randint(
                    int(augment_params["min_height_mult"] * height),
                    int(augment_params["max_height_mult"] * height),
                ),
                random.randint(
                    int(augment_params["min_width_mult"] * width),
                    int(augment_params["max_width_mult"] * width),
                ),
            )
        )
    box_yx = []
    for hw in box_hw:
        box_yx.append((random.randint(0, height - hw[0] - 1), random.randint(0, width - hw[1] - 1)))
    box_yx2 = box_yx[nbox:]
    box_hw2 = box_hw[nbox:]
    box_yx = box_yx[:nbox]
    box_hw = box_hw[:nbox]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV).astype(np.float)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV).astype(np.float)

    for i, yx in enumerate(box_yx):
        if random.random() < augment_params["sat_prob"]:
            mult = augment_params["sat_max"] - augment_params["sat_min"]
            sat_modifier = random.random() * mult + augment_params["sat_min"]
            hw = box_hw[i]
            img1[yx[0] : yx[0] + hw[0], yx[1] : yx[1] + hw[1], 1] *= sat_modifier
            yx2 = box_yx2[i]
            hw2 = box_hw2[i]
            img2[yx2[0] : yx2[0] + hw2[0], yx2[1] : yx2[1] + hw2[1], 1] *= sat_modifier
        if random.random() < augment_params["brightness_prob"]:
            mult = augment_params["brightness_max"] - augment_params["brightness_min"]
            brightness_modifier = random.random() * mult + augment_params["brightness_min"]
            hw = box_hw[i]
            img1[yx[0] : yx[0] + hw[0], yx[1] : yx[1] + hw[1], 2] *= brightness_modifier
            yx2 = box_yx2[i]
            hw2 = box_hw2[i]
            img2[yx2[0] : yx2[0] + hw2[0], yx2[1] : yx2[1] + hw2[1], 2] *= brightness_modifier

    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img1, img2


def flip_pair(
    img1,
    img2,
    bboxes,
    depth1,
    depth2
):
    img1, img2 = img2, img1
    depth1, depth2 = depth2, depth1
    return img1, img2, bboxes, depth1, depth2


def flip_lr(
    img1,
    img2,
    label,
    bboxes,
    depth1,
    depth2
):
    img1 = np.fliplr(img1).copy()
    img2 = np.fliplr(img2).copy()
    depth1 = np.fliplr(depth1).copy()
    depth2 = np.fliplr(depth2).copy()
    label = np.fliplr(label).copy()
    # TODO: fix flipping for bboxes
    #bboxes[:, [3, 1]] = 1 - bboxes[:, [1, 3]]
    return img1, img2, label, bboxes, depth1, depth2


def add_noise(
    img1, img2, noise_params: Dict[str, float]
):

    qe = np.random.uniform(noise_params["qe_low"], noise_params["qe_high"])
    bitdepth = noise_params["bit_depth"]
    baseline = noise_params["baseline"]
    sensitivity = np.random.uniform(
        noise_params["sensitivity_low"], noise_params["sensitivity_high"]
    )
    dark_noise = np.random.uniform(noise_params["dark_noise_low"], noise_params["dark_noise_high"])

    img1 = to_bayer(img1)
    img1 = np.sum(img1, axis=-1)
    img1 = add_camera_noise(
        img1,
        qe=qe,
        bitdepth=bitdepth,
        baseline=baseline,
        sensitivity=sensitivity,
        dark_noise=dark_noise,
    )
    img1 = _convert_GRGB_to_RGB(img1)

    img2 = to_bayer(img2)
    img2 = np.sum(img2, axis=-1)
    img2 = add_camera_noise(
        img2,
        qe=qe,
        bitdepth=bitdepth,
        baseline=baseline,
        sensitivity=sensitivity,
        dark_noise=dark_noise,
    )
    img2 = _convert_GRGB_to_RGB(img2)

    return img1, img2
