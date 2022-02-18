# INFO

I have removed all internal Standard refs from this repo, but I wanted to create a private repo before pushing it to public.

# Overview

This repository contains code to generate annotations and statistics for the StandardSim dataset. It also contains models and training/testing scripts for change detection and instance segmentation on the dataset. Finally it contains code to evaluate MiDaS monocular depth estimation on the dataset. 

The main entry point into the code is the synthetic_data_baselines folder. The folders and files discussed after Docker can be found within this folder.

# Docker

### Project template

To create a new project, first build cookiecutter docker image:
```bash
$ docker-compose build
```

Then run:
```bash
$ docker-compose run app ./start-project.py [flask|djangorestframework]
```

Available templates:
* django
* flask
* base-python

You may also pass in the directory of a template,

### Running tests

Build docker image:

```bash
$ docker-compose build
```

And run:

```bash
$ docker-compose run app ./run-tests
```

# StandardSim Dataset

## Annotations and Analysis

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

To train, validate and test the instance segmentation model (or any other model) run the above commands replacing train_changedetection.json with train_instanceseg.json, or with the name of the desired experiment.

## Configs

The configs folder contains json files with information for each experiment. Each is split into dictionaries with keys "train_dataset_params", "val_dataset_params", "test_dataset_params", "model_params", "visualization_params" and the parameters "printout_freq" and "model_update_freq" that are used directly in train.py.

For example, the breakdown of parameters in "train_dataset_params" could be

* "dataset_name": "simpleinstanceseg", # The name of the dataset
* "image_root": "/app/renders_multicam_diff_1", # The root directory path
* "json_root": "/app/synthetic_data_baselines/utils/instance_anno", # The path to the individual instance seg annotation files
* "loader_params":
	* "batch_size": 8, # Training batch size
    * "drop_last": true, # Whether to ignore the last items 
    * "pin_memory": true,
    * "num_workers": 8

* "mode": "train",
* "crop": false, # Use random crops
* "resize": false, # Randomly resize the data
* "spatial_resolution": [512, 360], # The resize resolution
* "overfit": false, # Overfit to 4 samples, useful for debugging purposes
* "normalize": true, # Normalize the images
* "augment": true, # Add data augmentation (noise, left-right flipping, etc.)
* "shuffle": true # Shuffle the samples over the epoch

and similarly for the change detection model parameters:

* "model_name": "changedetection", # Name of the model
* "num_classes": 4, # Number of classes in the labels (take, put, shift, no change)
* "dataset_name": "syntheticpairs", # Name of the dataset to use
* "rgb": true, # Use RGB inputs
* "depth": false, # Concatenate depth to RGB inputs
* "spatial_resolution": [512, 360], # Expected spatial resolution of input images
* "mode": "train", # Used to call .train() or .eval() on model
* "max_epochs": 4000, # Max number of epochs to train for
* "lr": 0.0004, # Learning rate to start
* "save_path": "/app/saved_models/changedetection_benchmark/", # Path to save models to
* "weights_path": "/app/saved_models/changedetection_benchmark/", # Path to load model weights from
* "load_weights": -1, # If -1, don't load weights. Otherwise specify epoch number to load.
* "val_epochs": [1950], # List of epochs to load weights from and evaluate
* "val_rate": 50, # How often to validate model in epochs
* "save_rate": 50, # How often to save model in epochs
* "lr_policy": "cosine", # Cosine decay. Could also use "step".
* "lr_decay_iters": 10, # Number of iterations to decay lr if using step lr decay.
* "batch_size": 16, 
* "loss_weights": true # Whether to use loss weights. The model file uses the dataset name to decide which to use.

Additionally in visualization params:

* "use_wandb": false,
* "save_images": true, # Whether to save images testing
* "save_path": "/app/synthetic_data_baselines/changedetection_test_predictions" # Path to save predictions to if previous flag set to true.

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
