
#!/usr/bin/env python3

import numpy as np
import os
from tqdm import tqdm
import time
import json

from pathlib import Path
import h5py

from typing import Dict

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat import Config, logger, Env
from habitat_baselines.common.environments import get_env_class
from habitat.datasets.registration import make_dataset

class NumpyArrayEncoder(json.JSONEncoder):
    r""" A simple Encoder to json dump a dictionary of np.ndarrys 
    """
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class Expert_Model:
    r"""Defines the expert(shortest path follower) whose experience is used to train a 
        policy thorough Behvaioral Cloning (BC), the experience is recorded to an external 
        dataset which is then stored in the HDF5 binary data format.
        
        :param config: habitat.Config object. Config file for running the behavioral cloning 
            algorithm.
        :param env: habitat.Env object. The environment which the expert interacts with
    """

    def __init__(self, config:Config, env:Env):
        self.env = env
        self.config = config

        self.goal_radius = self.env.episodes[0].goals[0].radius
        if self.goal_radius is None:
            self.goal_radius = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE

        self.expert = ShortestPathFollower(self.env.habitat_env.sim, self.goal_radius, False)
        self.expert.mode = config.IMITATION.EXPERT.mode
    
    @staticmethod
    def sanity_check(path:str) -> None:
        r"""A simple function to checks if the necessary directories exist and creates them 
        if they don't
        
        :param path: String object. A check is carried out on this path to check if 
            the necessary directories exist. If they don't, the directories are created    
        :return: None 
        """
        ref_path = Path(path)
        ref_path.mkdir(parents=True, exist_ok=True)
        return

    def store_many_hdf5(self, 
        expert_dict:dict, 
        path:str
    ) -> None:
        r"""Stores a dictionary of expert trajectories to HDF5.
            
        :param expert_dict: Dictionary object. It comprises of the expert trajectories. 
            The keys of this dictionary is made up of actions, depth, pointgoal_with_gps-
            _compass, rgb, num_episodes, num_entries)
        :param path: String object. The path where the collected trajectories would be stored 
            at
        :return None 
        """
        num_images = expert_dict['num_entries']

        # Create a new HDF5 file
        with h5py.File(os.path.join(path, "expert_trajectories.h5"), "w") as f:

            #create a subgroup for the main set
            g = f.create_group('MainCourse')
            actions = g.create_dataset('actions', data=expert_dict['action'], compression = "gzip")
            depth = g.create_dataset('depth', data=expert_dict['depth'], compression = "gzip")
            pg = g.create_dataset('pointgoal_with_gps_compass', data=expert_dict['pointgoal_with_gps_compass'], compression = "gzip")
            images = g.create_dataset('rgb', data=expert_dict['rgb'], compression = "gzip")

            #create a subgroup for meta data of the dataset
            h = f.create_group('Sides')
            meta_dict = {
                'num_episodes': expert_dict['num_episodes'], 
                'num_entries': expert_dict['num_entries']
            }

            meta = h.create_dataset('metadata', data=json.dumps(meta_dict, cls = NumpyArrayEncoder))
            del meta_dict

         return

    def generate_expert_trajectory(self, 
        phase:str, 
        iterations:int, 
        path:str, 
        logger:logger
    ) -> None:
        r"""Generates a record of the actions taken by an expert when this expert is allowed 
        to run in the specified environment for the required number of trajectories.
            
        :param phase: String object. This variable takes one of the two values ("train", "val") depending on 
            the setting for which the expert trajectories are being collected
        :param iterations: Integer. Number of episodes/trajectories for which the expert navigates 
            the environment 
        :param path: String object. A format string that gets substituted by the phase. The expert 
            trajectories are then stored at this path       
        :param logger: habitat.logger object. The logger to record the progress of the collection of 
            the expert trajectories 
        :return None
        """
        start_time = time.time()
        ep_list = []

        #initialize lists
        actions_list = []
        
        #list that split the observations recorded 
        rgb_list = []
        pg_list = []
        depth_list = []
        
        ep_idx = 0
        
        #this list is used to demarcate episodes
        num_entries = 0

        logger.info ("Running the generation cycle")
        logger.info ("Number of episodes used in this iteration {}".format(iterations))
        
        #create a dataset of the number of trajectories
        for ep_idx in tqdm(range(iterations)):
            #begins a new episode
            obs = self.env.reset()

            while not self.env.habitat_env.episode_over:
                rgb = obs['rgb']
                depth = obs['depth']
                pg = obs['pointgoal_with_gps_compass']

                best_action = self.expert.get_next_action(
                     self.env.habitat_env.current_episode.goals[0].position
                     )

                #running the episode
                if best_action is not None:
                    
                    rgb_list.append(rgb)
                    depth_list.append(depth)
                    pg_list.append(pg)

                    actions_list.append(best_action)

                    obs, reward, done, _ = self.env.step(action=best_action)
                    
                #either of this being true indicates completion of the episode
                if done or best_action is None:
                    ep_idx += 1
                    break

        #convert to numpy arrays
        logger.info("Consolidating the expert trajectories collected")

        #conversion to numpy arrays to save to an npz archive
        action_list = np.array(actions_list)
        rgb_list = np.array(rgb_list)
        depth_list = np.array(depth_list)
        pg_list = np.array(pg_list)

        num_episodes = np.array(iterations)

        assert (len(rgb_list) == len(actions_list) and 
               len(depth_list) == len(actions_list)), "The number of observations and the actions do not match"

        num_entries = np.array(len(rgb_list))

        if (phase == 'train'): logger.info("Building a train dataset of expert trajectories")
        elif (phase == 'val'): logger.info("Building a val dataset of expert trajectories ")

        expert_dataset = {
            'action': action_list,
            'depth': depth_list,
            'pointgoal_with_gps_compass': pg_list,
            'rgb': rgb_list,
            'num_episodes': num_episodes,
            'num_entries': num_entries,  
        }

        self.store_many_hdf5(expert_dataset, path.format(split=phase))

        logger.info("Recording the required number of trajectories took {} seconds".
                            format(time.time()-start_time))
        

        return