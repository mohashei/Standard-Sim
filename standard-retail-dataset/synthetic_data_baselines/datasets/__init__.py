"""This package includes all the modules related to data loading and preprocessing.
    To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
"""
import importlib
import torch
from torch.utils import data
from synthetic_data_baselines.datasets.base_dataset import BaseDataset

class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_inputs = next(self.loader)
        except StopIteration:
            self.next_inputs = None
            return
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        with torch.cuda.stream(self.stream):
            for i in range(len(self.next_inputs)):
                self.next_inputs[i] = self.next_inputs[i].cuda(non_blocking=True)
                self.next_inputs[i] = self.next_inputs[i].float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_inputs is None:
            return self.next_inputs
        inputs = []
        for i in range(len(self.next_inputs)):
            inputs.append(self.next_inputs[i])
            if inputs[i] is not None:
                inputs[i].record_stream(torch.cuda.current_stream())
        self.preload()
        return inputs


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError('In {0}.py, there should be a subclass of BaseDataset with class name that matches {1} in lowercase.'.format(dataset_filename, target_dataset_name))

    return dataset


def create_dataset(configuration):
    """Create a dataset given the configuration (loaded from the json file).
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and train.py/validate.py
    Example:
        from datasets import create_dataset
        dataset = create_dataset(configuration)
    """
    data_loader = CustomDatasetDataLoader(configuration)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading
        according to the configuration.
    """
    def __init__(self, configuration):
        self.configuration = configuration
        dataset_class = find_dataset_using_name(configuration['dataset_name'])
        self.dataset = dataset_class(configuration)
        print("dataset [{0}] was created".format(type(self.dataset).__name__))

        # if we use custom collation, define it as a staticmethod in the dataset class
        custom_collate_fn = getattr(self.dataset, "collate_fn", None)
        
        if self.configuration['is_multigpu']:
            self.dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=configuration["shuffle"])
        else:
            self.dist_sampler = None
        
        if callable(custom_collate_fn):
            self.dataloader = data.DataLoader(self.dataset, **configuration['loader_params'], shuffle=self.dist_sampler==None, collate_fn=custom_collate_fn, sampler=self.dist_sampler)
        else:
            self.dataloader = data.DataLoader(self.dataset, **configuration['loader_params'], shuffle=self.dist_sampler==None, sampler=self.dist_sampler)


    def load_data(self):
        return self

    def update_sampler(self, epoch):
        if self.configuration['is_multigpu']:
            self.dist_sampler.set_epoch(epoch)

    def get_custom_dataloader(self, custom_configuration):
        """Get a custom dataloader (e.g. for exporting the model).
            This dataloader may use different configurations than the
            default train_dataloader and val_dataloader.
        """
        custom_collate_fn = getattr(self.dataset, "collate_fn", None)
        if callable(custom_collate_fn):
            custom_dataloader = data.DataLoader(self.dataset, **self.configuration['loader_params'], collate_fn=custom_collate_fn)
        else:
            custom_dataloader = data.DataLoader(self.dataset, **self.configuration['loader_params'])
        return custom_dataloader
    

    def __len__(self):
        """Return the number of data in the dataset.
        """
        return len(self.dataset)


    def __iter__(self):
        """Return a batch of data.
        """
        for data in self.dataloader:
            yield data