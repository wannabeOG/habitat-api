#!/usr/bin/env python3

import torch
from torch.utils.data import IterableDataset

import numpy as np

import h5py
import os
import json
import random
import collections
from typing import List

class ExpertDataset(IterableDataset):
    r"""An Iterable dataset over the train and val folders created by the Expert agent
    
    :param path_to_dataset: A string object. The path of either the train/val folders
    :param batch_size: Integer. The batch size with which the dataloaders over the dataset 
        are instantiated with. 
        
    Instead of loading the entire dataset into the memory which could be an expensive affair,
    a (mult_factor * batch_size) number of observation-action pairs are preloaded into the 
    memory. This mult_factor is calculated as shown below:

    mult_factor = GIF(log2(number of entries in the dataset // batch_size))
    log2: logarithm with a base 2
    GIF: Greatest Integer Function
    """
    def __init__(self, 
        path_to_dataset:str, 
        batch_size:int,
    ):
        super().__init__()
        self.path_to_dataset = path_to_dataset
        #preloads (preload_size) number of (observations, actions) from the file
        self._preload = []
        self.batch_size = batch_size
       
        with h5py.File(self.path_to_dataset, 'r') as f:
            metadata = json.loads(f['Sides/metadata'][()])
            self.num_episodes = metadata['num_episodes']
            self.entries = metadata['num_entries']

        if (self.entries < self.batch_size):
            self.batch_size = np.power(int(np.log2(self.entries)), 2)
            mult_factor = 1

        elif((self.entries // self.batch_size) <= 2): mult_factor = 1
        else : mult_factor = int(np.log2(self.entries//self.batch_size))
        self.preload_size = self.batch_size * mult_factor

        self.cau_iteration_num = self.entries % self.preload_size
        self.cau_start = self.preload_size * (self.entries // self.preload_size)

    def block_shuffle(self, 
        lst:list, 
        block_size:int
    ) -> List:
        r"""Randomly shuffles a list in blocks

        :param lst: List object. A list of all the possible indices that are relevant to the 
            dataset instance at hand. For example if a dataset instance has 512 entries; the
            list contains indices from 0 to 511.
        :param block_size: Integer. This is set to the preload_size as preload_size number of 
            entries are loaded into the memory at a time
        :return list: A list of block shuffled indices

        This block shuffling ensures that selecting indices always results in a contigous set 
        of indices being selected any time.  
        """
        blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
        random.shuffle(blocks)

        return [item for block in blocks for item in block]

    def _load_next(self):
        r"""Reads in a preload_size number of entries from the HDF5 dataset. Once read, these
        entries are stored in a list and retuned one at time when this class method is called.  
        """
        if len(self._preload) == 0:
            
            #terminate condition
            if (len(self.indices_list) == 0):
                raise StopIteration
            
            with h5py.File(self.path_to_dataset,'r') as f: 
                #loop varibles
                temp_dict = dict()
                #print ("Indices list", self.indices_list)
                slicing_start = self.indices_list[0]
                
                if (slicing_start == self.cau_start):
                    slicing_end = self.indices_list[self.cau_iteration_num-1]
                    iteration_num = self.cau_iteration_num
                else:
                    slicing_end = self.indices_list[self.preload_size - 1]
                    iteration_num = self.preload_size

                del self.indices_list[:iteration_num]
                
                #load a preload_size of observations and actions into memory
                for key in f['MainCourse']:
                    temp_dict[key] = f['MainCourse/'+ key][()][slicing_start:slicing_end + 1]

                #precautionary measure to sort the keys of the dictionary alphabetically
                temp_dict = collections.OrderedDict(sorted(temp_dict.items()))

                self._preload = [[temp_dict[key][idx] for key in temp_dict.keys()] 
                                 for idx in range(iteration_num)]

                #make the variables fall out of scope
                del temp_dict
                del slicing_start
                del slicing_end
                del iteration_num

                #breaks any temporal dependancy between the individual scenes in the created list
                random.shuffle(self._preload)
                
        return self._preload.pop()

    def __next__(self):
        action, depth, pg_compass, rgb = self._load_next()

        #convert np.ndarrays to tensors
        action = torch.tensor(action)
        depth = torch.from_numpy(depth)
        pg_compass = torch.from_numpy(pg_compass)
        rgb = torch.from_numpy(rgb)

        return {
            'action': action,
            'depth': depth,
            'pointgoal_with_gps_compass': pg_compass,
            'rgb': rgb 
        }
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.entries
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        self.indices_list = list(
            self.block_shuffle(list(range(start, end)), self.preload_size)
        )

        return self 
