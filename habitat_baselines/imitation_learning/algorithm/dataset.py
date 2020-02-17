#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset

import habitat
import numpy as np
from skimage import io

class ExpertDataset(Dataset):
    r"""Dataset class over the train and val folders created by the Expert agent

    Args:
        path_to_dataset: The path to either the train/val split of the expert trajectories
        num_trajectories: Number of episodes that will be used in the training/validation process.
                        Default value uses all the trajectories available in the dataset
    """

    def __init__(self, path_to_dataset:str, num_trajectories:int = -1):
        self.dataset = np.load(path_to_dataset)
    
        #number of trajectories in the dataset
        dataset_trajectories = self.dataset['num_episodes'].item()

        #this flag is set when the images have been recorded in the dict and have not been saved to the disk
        self.record_images = isinstance(self.dataset['rgb'][0], np.ndarray) 

        self.num_trajectories = num_trajectories
        assert (self.num_trajectories <= dataset_trajectories)\
        , "The expert has only recorded {} trajectories".format(dataset_trajectories)
        
        if (num_trajectories == -1): self.num_trajectories = dataset_trajectories
            
        #use lesser number of trajectories than recorded
        list_of_indices = np.where(self.dataset['episode_pointer'])[0]

        #slicing indice records where the dataset needs to be demarcated
        self.slicing_indice = list_of_indices[self.num_trajectories-1] - (self.num_trajectories-1)
    
        #sanity checks
        assert(len(self.dataset['rgb'][:self.slicing_indice]) == len(self.dataset['action'][:self.slicing_indice]))\
        ,"Check your Expert dataset, images array and the action array don't match"
        assert(len(self.dataset['depth'][:self.slicing_indice]) == len(self.dataset['action'][:self.slicing_indice]))\
        ,"Check your Expert dataset, depth array and the action array don't match"


    def __len__(self):
        return len(self.dataset['rgb'][:self.slicing_indice])


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        assert (idx < self.slicing_indice), "This index out of range"
        
        path_to_image = self.dataset['rgb'][idx]

        #the references have been saved to dict
        if (not self.record_images):
            rgb = io.imread(path_to_image)
        else:
            rgb = self.dataset['rgb'][idx]

        #convert it to (C X H X W) format for compatibility with Torch
        rgb = torch.tensor(rgb)
        
        depth = torch.tensor(self.dataset['depth'][idx])
        action = torch.tensor(self.dataset['action'][idx])
        pg = torch.tensor(self.dataset['pointgoal_with_gps_compass'][idx])
        
        sample = {'rgb': rgb, 'depth': depth, 'action':action, 'pointgoal_with_gps_compass':pg}
        return sample
