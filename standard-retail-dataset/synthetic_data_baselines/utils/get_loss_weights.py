""" Functions to calculate: dataset image mean and std for normalization; loss weights for class imbalance.
"""
import logging

import torch

from synthetic_data_baselines.datasets.syntheticpairs_dataset import SyntheticPairsDataSet

logger = logging.getLogger(__name__)


class StatisticParams:
    def __init__(self):
        super().__init__()
        self.config = {
            "dataset_name": "syntheticpairs",
            "root": "/app/renders_multicam_diff_1",
            "mode": "train",
            "crop": False,
            "resize": False,
            "spatial_resolution": [512, 360],
            "overfit": False,
            "normalize": True,
            "augment": False
        }


def loss_weights(params: StatisticParams):
    mean = 0.0
    std = 0.0
    nb_samples = 0.0

    lb_bg, lb_fg, tot = 0, 0, 0

    # Assumes data loaders return images as arrays
    loader = torch.utils.data.DataLoader(
        SyntheticPairsDataSet(params.config),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    class_totals = [0, 0, 0, 0]
    i = 0
    for i, data in enumerate(loader):
        print(i)
        img1, img2, label, _, _, scene, _, _ = data
        data = img1.view(1, img1.size(1), -1)
        data = data.float()
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += 1

        data = img2.view(1, img2.size(1), -1)
        data = data.float()
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += 1

        for j in range(len(class_totals)):
            class_totals[j] += torch.nonzero(label == j).shape[0]

        i += 1
        if i % 1000 == 0:
            logger.info(f"{i} done.")

    logger.info("Total number samples ", nb_samples)
    mean /= nb_samples
    std /= nb_samples
    num_px = sum(class_totals)
    for j in range(len(class_totals)):
        class_totals[j] /= num_px
        
    #logger.info("Mean and standard dev ", mean, std)
    print("Loss weights ", [(1 - class_totals[0]) / 3, (1 - class_totals[1]) / 3, (1 - class_totals[2]) / 3, (1 - class_totals[3]) / 3])
  
def depth_stats(params: StatisticParams):
    mean, std = 0.0, 0.0
    nb_samples = 0.0

    # Assumes data loaders return images as arrays
    loader = torch.utils.data.DataLoader(
        SyntheticPairsDataSet(params.config),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    print(len(loader))
    for i, data in enumerate(loader):
        print(i)
        _, _, label, _, _, scene, depth1, depth2 = data
        data = depth1[0].float()
        mean += data.mean()
        std += data.std()
        nb_samples += 1

        data = depth2[0].float()
        mean += data.mean()
        std += data.std()
        nb_samples += 1

        if i % 1000 == 0:
            logger.info(f"{i} done.")

    logger.info("Total number samples ", nb_samples)
    mean /= nb_samples
    std /= nb_samples
    print("mean ", mean)
    print("std ", std)    
    

if __name__ == "__main__":
    p = StatisticParams.parse()
    loss_weights(p)
    #depth_stats(p)


