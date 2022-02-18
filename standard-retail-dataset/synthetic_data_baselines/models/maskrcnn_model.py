import os
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.metrics import jaccard_score

import torch
import torchvision

from .base_model import BaseModel
from .utils import reduce_dict
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#from torchvision.models.detection import FasterRCNN
#from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNNModel(BaseModel):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.configuration = configuration
        self.loss_names = ['total']
        self.network_names = ['instanceseg']
        self.num_classes = configuration["num_classes"]

        '''
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        bbox_in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(bbox_in_features, self.num_classes)
        model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, self.num_classes, kernel_size=(1,1))
        '''

        '''
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        backbone = SiameseNet(backbone=backbone)
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
        model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
        '''
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           self.num_classes)

        self.netinstanceseg = model
        self.model = self.netinstanceseg
        
        # Check number of parameters in model
        print("Number total params ", sum(p.numel() for p in self.netinstanceseg.parameters() if p.requires_grad))

        self.netinstanceseg = self.netinstanceseg.cuda()

        # storing predictions and labels for validation
        w, h = self.configuration["spatial_resolution"]
        self.val_predictions = torch.zeros((2*949, h, w)) # number samples is 949 which makes 2*949 images
        self.val_labels = torch.zeros((2*949, h, w)) 
        self.val_count = 0

    def init_train_mechanics(self):
        if self.is_train:  # only defined during training time
            # Loss definitions
            #self.criterion_loss = torch.nn.CrossEntropyLoss() # Segmentation output loss

            # Optimizer definitions
            main_optimizer = torch.optim.SGD(self.netinstanceseg.parameters(), lr=self.configuration["lr"])
            #optimizer = AdamW(self.netinstanceseg.parameters(), lr=1e-5 * batch_size / 4 * params.world_size, weight_decay=1e-5)
            self.optimizers = [main_optimizer]
        

    def set_input(self, inputs, mode='train'):
        """ Unpack input data from the dataloader and perform necessary pre-processing steps.
        Format of data is images (torch.Tensor), labels (torch.Tensor), img name (str)

        """
        img, target, scene = inputs
        img = [i.float().cuda() for i in img]
        print("img[0] shape ", img[0].shape)
        print("len img ", len(img))
        self.input = img
        
        target = [{k: v.cuda() for k, v in t.items()} for t in target]
        print("len target ", len(target))
        print("target[0] masks ", target[0]["masks"].shape)
        print("target[0] boxes ", target[0]["boxes"].shape)
        print("target[0] labels ", target[0]["labels"].shape)
        self.target = target
        self.name = scene

    def forward(self, mode):
        """Run forward pass.
        """
        self.output = self.netinstanceseg(self.input, self.target)

    def backward(self):
        """Calculate losses; called in every training iteration.
        """
        loss_total = sum(loss for loss in self.output.values())
        print("loss total device ", loss_total.get_device())
        print("self.output loss devices ", [l.get_device() for l in self.output.values()])
        loss_total.backward()

        loss_dict_reduced = reduce_dict(self.output)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        self.loss_total = losses_reduced.item()
        #torch.cuda.empty_cache()

    def optimize_parameters(self):
        """Calculate gradients and update network weights.
        """
        # calculate gradients
        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad()

    def test(self, save_images=False, output_path='/app/synthetic_data_baselines/predictions'):
        # run the forward pass
        with torch.no_grad():
            self.forward(mode='val')
        
        # During eval mode model outputs a list
        out = self.output[0]

        # Check if any object detected and make output all zeros if not
        if len(out['boxes']) == 0:
            mask_img = torch.zeros((self.input.shape[1], self.input.shape[2]))
        else:
            mask_img = torch.squeeze(torch.argmax(torch.squeeze(out['masks'], dim=0), dim=0), dim=0).cpu()
        
        self.val_predictions[self.val_count] = mask_img
        self.val_count+=1
        
        palette = [
            0, 0, 0,
            159, 4, 22,
            98, 190, 48,
            122, 130, 188
        ]

        # Save predictions
        if save_images:
            img1 = Image.fromarray(mask_img1.numpy().astype(np.uint8))
            img1.putpalette(palette * 64)
            img1.save(os.path.join(output_path, self.name+".png"))
            print("Saving ", os.path.join(output_path, self.name+".png"))

        self.input, self.label, self.output = [], [], []

    def pre_epoch_callback(self, epoch):
        #self.netinstanceseg = torch.jit.script(self.netinstanceseg)
        pass

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

