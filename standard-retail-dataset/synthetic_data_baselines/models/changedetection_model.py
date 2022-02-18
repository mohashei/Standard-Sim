import os
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.metrics import jaccard_score

import torch
from .singlestream import SingleStream
from .base_model import BaseModel

class ChangeDetectionModel(BaseModel):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.configuration = configuration
        self.loss_names = ['segmentation']
        self.network_names = ['changedetection']
        self.num_classes = configuration["num_classes"]

        self.netchangedetection = SingleStream(4, True, configuration["rgb"], configuration["depth"])
        self.model = self.netchangedetection

        # Check number of parameters in model
        print("Number total params ", sum(p.numel() for p in self.netchangedetection.parameters() if p.requires_grad))

        self.netchangedetection = self.netchangedetection.cuda()

        # storing predictions and labels for validation
        if self.configuration["dataset_name"] == "syntheticpairs" or self.configuration["dataset_name"] == "simplesyntheticpairs":
            w, h = self.configuration["spatial_resolution"]
            if self.configuration["mode"] == 'val':
                self.val_predictions = torch.zeros((949, h, w))
                self.val_labels = torch.zeros((949, h, w))
            elif self.configuration["mode"] == 'test':
                self.val_predictions = torch.zeros((955, h, w))
                self.val_labels = torch.zeros((955, h, w))
            self.val_count = 0
        else:
            print("Unknown dataset name in changedetection_model.py")

    def init_train_mechanics(self):
        if self.is_train:  # only defined during training time
            # Loss definitions
            loss_weights = torch.Tensor([0.0035252888617000786, 0.33161259399877635, 0.3316452266650099, 0.3332168904745137]).cuda()
            self.criterion_loss = torch.nn.CrossEntropyLoss(weight=loss_weights) # Segmentation output loss

            # Optimizer definitions
            main_optimizer = torch.optim.SGD(self.netchangedetection.parameters(), lr=self.configuration["lr"])
            #optimizer = AdamW(self.netchangedetection.parameters(), lr=1e-5 * batch_size / 4 * params.world_size, weight_decay=1e-5)
            self.optimizers = [main_optimizer]
        

    def set_input(self, inputs, mode='train'):
        """ Unpack input data from the dataloader and perform necessary pre-processing steps.
        Format of data is images (torch.Tensor), labels (torch.Tensor), img name (str), domain (torch.Tensor), [original height dim (int), original width dim (int)], dataset name (str) 
        
        Note: When video is loaded for training a clip is randomly selected.
        When loaded during inference/evaluation the whole clip is returned as a list of temporal_resolution-length tensors.
        Assumes batch size 1 during inference
        """
        img1, img2, label, bboxes, classes, scene, depth1, depth2 = inputs
        img1 = img1.float().cuda()
        img2 = img2.float().cuda()
        depth1 = depth1.float().cuda()
        depth2 = depth2.float().cuda()
        self.input = (img1, img2, depth1, depth2)

        label = label.type(torch.LongTensor)
        self.label = label.cuda()

        self.name = scene

    def forward(self, mode):
        """Run forward pass.
        """
        self.output = self.netchangedetection(self.input)

    def backward(self):
        """Calculate losses; called in every training iteration.
        """
        self.loss_segmentation = self.criterion_loss(self.output, self.label)

    def optimize_parameters(self):
        """Calculate gradients and update network weights.
        """
        # calculate gradients
        self.loss_segmentation.backward()
        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad()

    def test(self, save_images=False, output_path='/app/synthetic_data_baselines/predictions'):
        # run the forward pass
        with torch.no_grad():
            self.forward(mode='val')
        self.val_labels[self.val_count] = self.label.cpu()
        out = torch.argmax(self.output, dim=1).cpu()
        self.val_predictions[self.val_count] = out

        palette = [
            0, 0, 0,
            159, 4, 22,
            98, 190, 48,
            122, 130, 188
        ]

        # Save predictions
        if save_images:
            img1 = Image.fromarray(torch.squeeze(out, dim=0).numpy().astype(np.uint8))
            img1.putpalette(palette * 64)
            img1.save(os.path.join(output_path, self.name[0]+"_pred.png"))
            print("Saving ", os.path.join(output_path, self.name[0]+"_pred.png"))

        self.input, self.label, self.output = [], [], []
        self.val_count+=1


    def post_epoch_callback(self, epoch, visualizer=None):
        # Calculate and show accuracy
        iou = jaccard_score(self.val_labels.flatten(), self.val_predictions.flatten(), average=None)
        
        metrics = OrderedDict()
        metrics['iou'] = iou

        if visualizer:
            visualizer.plot_current_validation_metrics(epoch, metrics)
        print(iou)
        print(np.mean(iou))

        # Re-initialize val buffers after evaluation
        w, h = self.configuration["spatial_resolution"]
        self.val_predictions = torch.zeros((949, h, w))
        self.val_labels = torch.zeros((949, h, w))
        self.val_count = 0

        return np.mean(iou), np.mean(iou[1:])

