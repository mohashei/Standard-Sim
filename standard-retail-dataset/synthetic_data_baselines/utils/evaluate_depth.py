"""
Functions to evaluate depth predictions

Fraction of pixels in ground truth equal to 0: 0.0008877665675593257

Performance of MiDaS on test set
Absolute relative error: 83.5923
Absolute difference: 10.0259
Square relative error: 625.7967

"""
import os
import imageio
import numpy as np
from pypfm import PFMLoader

import torch

def compute_errors(gt, pred, print_res=False):
    """Evaluates depth metrics"""

    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.shape[0]

    res_list = []
    for current_gt, current_pred in zip(gt, pred):
        valid_gt = current_gt
        valid_pred = current_pred
        
        nonzero_mask = (valid_gt > 0).bool()
        valid_gt = torch.masked_select(valid_gt, nonzero_mask)
        valid_pred = torch.masked_select(valid_pred, nonzero_mask)

        if len(valid_gt) == 0:
            continue
        else:
            thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
            a1 += (thresh < 1.25).float().mean()
            a2 += (thresh < 1.25 ** 2).float().mean()
            a3 += (thresh < 1.25 ** 3).float().mean()

            abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
            if print_res:
                res_list.append(torch.mean(torch.abs(valid_gt - valid_pred)).item())
            abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

            sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    if print_res:
        print(res_list)

    return [metric / batch_size for metric in [abs_rel, abs_diff, sq_rel, a1, a2, a3]]

if __name__ == "__main__":
    pred_root = '/app/MiDaS/output'
    gt_root = '/app/renders_multicam_diff_1'

    pfm_images = [f for f in os.listdir(pred_root) if f.endswith(".pfm")]
    gt_images = [f[:-4]+'-depth0001.exr' for f in pfm_images]
    loader = PFMLoader(color=False, compress=False)

    preds = []
    gts = []
    count_zeros = []
    count_zeros_percentages = []
    for i in range(len(pfm_images)):
        pfm_img = loader.load_pfm(os.path.join(pred_root, pfm_images[i]))
        pfm_img[pfm_img>100] = -1
        pfm_img[pfm_img==-1] = np.max(pfm_img)
        preds.append(pfm_img)
        gt_img = np.array(imageio.imread(os.path.join(gt_root, gt_images[i]), format="EXR-FI"))
        if gt_img.shape != (720, 1280):
            gt_img = gt_img[:,:,0]
        if 0 in gt_img:
            count_zeros_percentages.append(len(np.where(gt_img == 0)[0]) / 921600)
            count_zeros.append(gt_images[i])
        gt_img[gt_img>100] = -1
        gt_img[gt_img==-1] = np.max(gt_img)
        gts.append(gt_img)
    
    print("Num gt images with zeros ", len(count_zeros))
    print("Fraction of pixels equal zero ", np.mean(np.array(count_zeros_percentages)))
    print(compute_errors(torch.from_numpy(np.array(preds)), torch.from_numpy(np.array(gts))))



