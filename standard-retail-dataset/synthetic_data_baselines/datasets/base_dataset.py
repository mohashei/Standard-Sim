"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
"""
from abc import ABC, abstractmethod
import cv2
import numpy as np
import torch.utils.data as data


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    """

    def __init__(self, configuration):
        """Initialize the class; save the configuration in the class.
        """
        self.configuration = configuration

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point (usually data and labels in
            a supervised setting).
        """
        pass

    def pre_epoch_callback(self, epoch):
        """Callback to be called before every epoch.
        """
        pass

    def post_epoch_callback(self, epoch):
        """Callback to be called after every epoch.
        """
        pass