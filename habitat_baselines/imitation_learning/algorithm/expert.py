#!/usr/bin/env python3

import numpy as np
import os
from tqdm import tqdm
import time
import cv2
from pathlib import Path

from typing import Dict

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat import Config, logger
from habitat.datasets.registration import make_dataset

from habitat_baselines.common.environments import get_env_class


class Expert_Model:
    r"""Defines the expert(shortest path follower) whose experience is used to train a policy 
    thorough Behvaioral Cloning (BC), the experience is recorded in an external npz archive.

    Args: 
        config: Config file
    """

    def __init__(self, config:Config):
        self.config = config
        dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, 
                                config = config.TASK_CONFIG.DATASET)
        
        env_class = get_env_class(config.ENV_NAME)
        self.env = env_class(config=config, dataset=dataset)

        self.goal_radius = self.env.episodes[0].goals[0].radius
        if self.goal_radius is None:
            self.goal_radius = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE

        self.expert = ShortestPathFollower(self.env.habitat_env.sim, self.goal_radius, False)
        self.expert.mode = self.config.IMITATION.EXPERT.mode

    @staticmethod
    def sanity_check(path):
        r"""A simple function to checks if the necessary directories exist and creates them 
        if they don't

        Args:
            path: A string object. A check is carried on the parent of this specified directory path 
            and it is created if it doesn't exist

        Returns:
            None 
        """
        ref_path = Path(path)
        ref_path.mkdir(parents=True, exist_ok=True)


    def generate_expert_trajectory(self) -> None:
        r"""Generates a record of the actions taken by an expert when this expert is allowed 
        to run in the specified environment for the required number of trajectories.
        
        Args:
            num_episodes: number of episodes/trajectories for which the expert navigates the environment
            record_images: 
                If True:the images will also be saved in the numpy dict as it is. This would lead to more 
                memory consumption when the dataset would be loaded  
                If False: images will be saved to disk and only a reference will be recorded in the
                numpy dict
            flag: 
                If True: the training set would be created 
                If False: the validation set will be created
        
        Returns:
            None
        """

        start_time = time.time()
        expert_config = self.config.IMITATION.EXPERT
        image_extension = ".png"
        ep_list = []
        path_list = []

        num_episodes = expert_config.num_episodes
        split_dataset = expert_config.split_dataset
        record_images = expert_config.record_images

        training_episodes = int(split_dataset*num_episodes)
        ep_list.append(training_episodes)

        validation_episodes = num_episodes - training_episodes
        if(validation_episodes != 0): ep_list.append(validation_episodes)

        logger.add_filehandler(expert_config.log_file)

        #create the necessary directories to house the images
        if(not record_images):
            #training dataset is being created
            #str object
            image_path_train = os.path.join(Path(expert_config.train_path).parent, "images")
            self.sanity_check(image_path_train)
            path_list.append(image_path_train)

            if(split_dataset != 0.0):
                #str object
                image_path_val = os.path.join(Path(expert_config.val_path).parent, "images")
                self.sanity_check(image_path_val)
                path_list.append(image_path_val)

        #loop over the train/val sets
        for idx, num_episodes in enumerate(ep_list):

            #initialize lists
            actions_list = []
            rgb_list = []
            pg_list = []
            depth_list = []
            
            ep_idx = 0
            total_reward = 0.0
            
            #im_idx records the number of images 
            im_idx = 0

            #this list is used to demarcate episodes
            episode_pointer = []

            if(idx == 0): logger.info("Creating the training set")
            elif(idx == 1): logger.info("Creating the validation set")

            logger.info ("Environment creation successful")
            logger.info ("Agent stepping around the environment")

            for ep_idx in tqdm(range(num_episodes)):
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
                        
                        if (not record_images):
                            file_name = str(im_idx) + image_extension
                            
                            if(idx == 0):
                                path_to_store = os.path.join(image_path_train, file_name)
                            elif(idx == 1):
                                path_to_store = os.path.join(image_path_val, file_name)

                            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(path_to_store, rgb)
                            rgb_list.append(path_to_store)
                        
                        else:
                            rgb_list.append(rgb)
                        
                        depth_list.append(depth)
                        pg_list.append(pg)

                        #the corresponding observation
                        episode_pointer.append(False)

                        #pg_list.append(pg)
                        actions_list.append(best_action)

                        im_idx = im_idx + 1
                        obs, reward, done, _ = self.env.step(action=best_action)
                        total_reward += reward

                    #either of this being true indicates completion of the episode
                    if done or best_action is None:
                        episode_pointer.append(True)
                        total_reward = 0.0
                        ep_idx += 1
                        break

            #convert to numpy arrays
            logger.info("Consolidating the expert trajectories collected")
            
            #conversion to numpy arrays to save to an npz archive
            action_list = np.array(actions_list)
            rgb_list = np.array(rgb_list)
            depth_list = np.array(depth_list)
            pg_list = np.array(pg_list)

            episode_pointer = np.array(episode_pointer)
            num_episodes = np.array(num_episodes)
            
            assert (len(rgb_list) == len(actions_list) and 
                   len(depth_list) == len(actions_list)), "The number of observations and the actions do not match"


            if (idx == 0): logger.info("Building a train dataset of expert trajectories")
            elif (idx == 1): logger.info("Building a val dataset of expert trajectories ")

            expert_dataset = {
                'action': action_list,
                'rgb': rgb_list,
                'depth': depth_list,
                'pointgoal_with_gps_compass': pg_list,
                'num_episodes': num_episodes,
                'episode_pointer': episode_pointer,  
            }

            if(idx == 0):
                save_path = expert_config.train_path
            elif(idx == 1):
                save_path = expert_config.val_path

            np.savez(save_path, **expert_dataset)
                
        
        logger.info("Recording the required number of trajectories took {} seconds".
                            format(time.time()-start_time))




      