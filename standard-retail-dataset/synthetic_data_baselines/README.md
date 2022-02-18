# Overview

This repository contains code to generate annotations and statistics for the StandardSim dataset. It also contains models and training/testing scripts for change detection and instance segmentation on the dataset. Finally it contains code to evaluate MiDaS monocular depth estimation on the dataset.

# StandardSim Dataset

The standardsim folder contains scripts related to generating annotations and visualizing the dataset. The most important scripts are:

* analyze.py: Contains functions to generate dataset statistics such as number of stores per split, etc. Also contains a function to generate standalone depth maps.
* create_dataset_json.py: Creates a json file "synthetic_anno.json" that contains COCO-style annotations for the dataset. It also has useful functions that convert masks to Polygons and vice versa.
* create_instanceseg_anno.py: Creates a json file containing COCO-style instance segmentation masks for each sample in the dataset.
* synthetic_dataset_tools.py: A class to visualize the dataset that is used in the dataset SyntheticPairsDataSet; based on COCO demos.

The following scripts may also come in useful:

* check_imageset_lists.py: Script for checking if any scenes with rendering errors are in the dataroot folder. Also checks that all scenes in the image sets exist in the dataroot folder.
* check_outlier_samples.py: Script for calculating the average and standard dev of pixels in dataset images. Uses this to check what samples lie far from the mean.
* create_instanceseg_anno_dep.py: Creates a single json file to contain all instance segmentation annotations (Deprecated. Use create_instanceseg_anno.py).

There are two txt files that contain useful information:

* dataset_format.txt: A breakdown of the dataset annotation json file.
* scene_errors.txt: A list of scenes with rendering errors. Can be used to ensure these scenes are no longer in the dataset.

The following Jupyter Notebooks can be used as demos:

* synthetic_data_demo.ipynb: Visualizes the dataset samples and their annotations.

# Training and Testing

This repository supports training and testing for change detection and instance segmentation. The parameters for each experiment, including the model types, datasets and hyperparameters, are contained in json configuration files in the configs folder. Training and testing is run from the main directory.

## Usage

To train a change detection model run

python train.py --configfile configs/train_changedetection.json

To measure a model's performance on the validation set, run

python validate.py --configfile configs/train_changedetection.json

To test a change detection model run

python test.py --configfile configs/train_changedetection.json

## Configs

The configs folder contains json files with information for each experiment. 

## Datasets

The datasets folder contains dataset definitions that load data for different tasks. The files here are:

* augment.py: Contains functions to add noise, left-right flipping, and other augmentations to the data.
* base_dataset.py: Contains definition for BaseDataset which the other datasets subclass.
* simpleinstanceseg_dataset.py: Contains dataset for instance segmentation that does not use synthetic_anno.json to fetch annotations.
* simplesyntheticpairs_dataset.py: Contains dataset for change detection that does not use synthetic_anno.json to fetch annotations.
* syntheticpairs_dataset.py: Contains dataset for change detection that loads COCO-style annotations from synthetic_anno.json.
* utils.py: Contains function to convert change detection action (takes, puts and shifts) to an integer.

## Models

The models folder contains model class definitions for different tasks.

* base_model.py: Contains BaseModel, and Abstract Base Class, that all other models subclass for use with train.py.
* changedetection_model.py: Contains model definition for the change detection model with functions to format input properly for the model, save and evaluate outputs from the model.
* maskrcnn_model.py: Contains mask rcnn model definition for instance segmentation. Currently uses torchvision version.
* singlestream.py: Contains forward definition for change detection model with single encoder for rgb and depth.
* twostream.py: Contains forward definition for change detection model with separate encoders for rgb and depth.

The following files contain functions and models that replicate the torchvision's mask rcnn:

* maskrcnn.py
* resnet.py
* roi_head.py
* transform.py

There are also lots of helper functions for this in utils.py.

## Utils

A useful script is save_labels.py which has a function to save the change detection labels in color for easy visualization. It also has a script to save depth as a png for easier visualization.

## MiDaS 

To print a quantitative evaluation of MiDaS's outputs on StandardSim, run the script evaluate_depth.py in the utils folder with "python evaluate_depth.py". The script contains file paths that may need to be changed.
