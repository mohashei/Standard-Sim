import os
import time
import argparse

from models import create_model
from datasets import create_dataset
from utils import parse_configuration

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

"""Performs training of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
    export: Whether to export the final model (default=True).
"""
def train(rank, config_file, world_size):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    is_multigpu = False
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1 and world_size > 1:
            is_multigpu = True
            print(f"Running multi-GPU setup for GPU {rank}")
            setup(rank, world_size)
            torch.cuda.set_device(rank)

    configuration = parse_configuration(config_file)
    if rank == 0 and configuration['visualization_params']['use_wandb']:
        import wandb
        wandb.init(project="synthetic-release-baselines")

    configuration['train_dataset_params']['is_multigpu'] = is_multigpu
    configuration['val_dataset_params']['is_multigpu'] = is_multigpu
    train_dataset = create_dataset(configuration['train_dataset_params']) # Takes 34 seconds
    train_dataset_size = len(train_dataset)

    val_dataset = create_dataset(configuration['val_dataset_params']) # Takes 5 seconds
    val_dataset_size = len(val_dataset)

    model = create_model(configuration['model_params'])
    if is_multigpu:
        model.model = DDP(model.model, device_ids=[rank])
    else:
        model.model = nn.DataParallel(model.model)
    model.init_train_mechanics()
    model.setup()
    if rank == 0 and configuration['visualization_params']['use_wandb']:
        wandb.watch(model.model, log='all')

    starting_epoch = 0
    num_epochs = configuration['model_params']['max_epochs']
    j = 0
    for epoch in range(starting_epoch, num_epochs):
        epoch_start_time = time.time()  # timer for entire epoch
        train_dataset.dataset.pre_epoch_callback(epoch)
        model.pre_epoch_callback(epoch)
        train_iterations = len(train_dataset)
        train_batch_size = configuration['train_dataset_params']['loader_params']['batch_size']
        
        model.train()
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            print("i ", i)
            model.set_input(data, mode='train')   # unpack data from dataset and apply preprocessing
            
            model.forward(mode='train')
            model.backward()
            
            if i % configuration['model_update_freq'] == 0:
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                
            #if i % configuration['printout_freq'] == 0:
            #    losses = model.get_current_losses()

            #losses = model.get_current_losses()
            
            if rank == 0 and configuration['visualization_params']['use_wandb']:
                wandb.log({"Train Loss": losses["total"].item()})
        
        train_dataset.dataset.post_epoch_callback(epoch)

        if rank == 0:
            print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch, num_epochs, time.time() - epoch_start_time))
        
        if rank == 0 and j % configuration['model_params']['val_rate'] == 0 and j != 0:
            print("Validating model at epoch {0}".format(epoch))
            model.eval()
            model.pre_epoch_callback(epoch)
            for i, data in enumerate(val_dataset):
                model.set_input(data, 'val')
                # Run inference
                model.test()
            mean_iou, no_bckgrnd = model.post_epoch_callback(epoch)
            if rank == 0 and configuration['visualization_params']['use_wandb']:
                wandb.log({"Val": mean_iou, "No Background": no_bckgrnd})
                print("Val ", mean_iou, " No Background ", no_bckgrnd)
            model.train()

        if rank == 0 and j % configuration['model_params']['save_rate'] == 0:
            print('Saving model at the end of epoch {0}'.format(epoch))
            model.save_networks(epoch)
            model.save_optimizers(epoch)

        model.update_learning_rate() # update learning rates every epoch
        j+=1


    if is_multigpu:
        cleanup()
    
    return model.get_hyperparam_result()

def run_train(train_fn, args, world_size):
    mp.spawn(train_fn,
             args=(args, world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--configfile', help='path to the configfile')

    args = parser.parse_args()
    n_gpus = torch.cuda.device_count()
    #train(args.configfile)
    run_train(train, args.configfile, n_gpus)


