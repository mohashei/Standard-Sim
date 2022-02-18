import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import create_dataset
from utils import parse_configuration
from models import create_model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

"""Performs validation of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
"""
def validate(rank, config_file, world_size):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    is_multigpu = False
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1 and world_size > 1:
            is_multigpu = True
            print(f"Running multi-GPU setup for GPU {rank}")
            setup(rank, world_size)
            torch.cuda.set_device(rank)

    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    configuration['val_dataset_params']['is_multigpu'] = is_multigpu
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    val_epochs = configuration['model_params']['val_epochs']
    for epoch in val_epochs:
        configuration['model_params']['load_weights'] = epoch
        model = create_model(configuration['model_params'])
        if is_multigpu:
            model.model = DDP(model.model, device_ids=[rank])
        else:
            model.model = nn.DataParallel(model.model)
        model.init_train_mechanics()
        model.setup()
        model.eval()
        
        model.pre_epoch_callback(epoch)
        for i, data in enumerate(val_dataset):
            model.set_input(data, 'val')
            # Run inference
            model.test(save_images=configuration["visualization_params"]["save_images"],
                    output_path=configuration["visualization_params"]["save_path"])

        mean_iou, no_bckgrnd = model.post_epoch_callback(epoch)

    if is_multigpu:
        cleanup()

def run_val(val_fn, args, world_size):
    mp.spawn(val_fn,
             args=(args, world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('--configfile', help='path to the configfile')

    args = parser.parse_args()
    n_gpus = torch.cuda.device_count()
    run_val(validate, args.configfile, n_gpus)

