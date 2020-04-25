#!/usr/bin/env python3

import os
import time
from typing import Dict, List

import numpy as np
import time
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import torch.optim as optim

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video
)

from habitat_baselines.common.base_trainer import BaseTrainer, BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.imitation_learning.algorithm.dataset import ExpertDataset
from habitat_baselines.imitation_learning.algorithm.expert import Expert_Model
from habitat_baselines.imitation_learning.algorithm.policy import PointNavBaselinePolicy

from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat.datasets.registration import make_dataset

from habitat_baselines.common.tensorboard_utils import TensorboardWriter

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self

def collate_fn(batch):
    r""" A custom collate function used by the dataloaders defined over the iterable dataset
    """
    final_list = [[batch[idx][key] for idx in range(len(batch))] for key in batch[0].keys()]
    
    actions_batch = torch.stack(final_list[0], dim = 0)
    depth_batch = torch.stack(final_list[1], dim = 0)
    pointgoal_with_gps_compass_batch = torch.stack(final_list[2], dim = 0)
    rgb_batch = torch.stack(final_list[3], dim = 0)

    return ObservationsDict({
    'action': actions_batch, 
    'depth': depth_batch, 
    'pointgoal_with_gps_compass': pointgoal_with_gps_compass_batch,
    'rgb': rgb_batch
    })


@baseline_registry.register_trainer(name="bc")
class BCTrainer(BaseRLTrainer):
    r"""A Trainer class for Behavioral Cloning (BC)

    :param config: habitat.Config object. Config file for running the behavioral cloning 
        algorithm. 
    :param observation_space: A property of the environment in which the expert trajectories 
        had been recorded. The observation space records the type of observations unique to 
        the sensors that the expert had been allowed access to in the generation cycle.
    :param actionS_space: A property of the agent. The range of actions to which the policy 
        being learnt by the agent will be limited to. 
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self,config:Config):
        super().__init__(config)
        self.agent = None
        self.env = None
        self.expert = None
        self.config = config
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        logger.add_filehandler(self.config.LOG_FILE)
        if config is not None:
            logger.info(f"config: {config}")

    def _setup_agent(self, 
        config:Config,
    ) -> None:
        r"""Initializes an agent and a log file for the problem of PointGoal Navigation

        :param config: A config file with the relevant params for initializing the agent
        :return None
        """
        
        imitation_config = self.config.IMITATION

        self.agent = PointNavBaselinePolicy(
            observation_space = self.env.observation_space,
            action_space = self.env.action_space,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            hidden_size = imitation_config.BC.hidden_size
        )

        return 

    def _get_dataloaders_size_dataset(self, 
        bc_config:Config, 
        phase:str
    ) -> List:
        r"""Creates dataloaders over an instance of the ExpertDataset (initialized 
        using the phase). The dataloader and the size of this dataset are then returned 

        :param bc_config: config node with the relevant params for behavioral cloning 
        :param phase: This variable takes one of the two values ("train", "val") which depends 
            on the setting the BCtrainer is being operated in
        :return [dataloaders, size]
        """
        path = os.path.join(
                bc_config.EXPERT.dataset_path.format(split = phase), 
                "expert_trajectories.h5"
            )
        batch_size = bc_config.BC.batch_size
        
        #create instances of the ExpertDataset
        image_dataset = ExpertDataset(path, batch_size)
        size_of_dataset = image_dataset.entries

        if (image_dataset.batch_size != bc_config.BC.batch_size):
            logger.info("ERROR, batch size is bigger than the dataset\
                ,updating the batch_size to be {}".format(image_dataset.batch_size))
            batch_size = image_dataset.batch_size
        
        dataloader = DataLoader(image_dataset, 
                        batch_size=batch_size, 
                        num_workers=bc_config.BC.num_workers, 
                        pin_memory=bc_config.BC.pin_memory,
                        collate_fn = collate_fn
                    )
                
        return [dataloader, size_of_dataset]

    @staticmethod
    def _exp_lr_scheduler(optimizer, 
        epoch, 
        init_lr=0.0008, 
        lr_decay_epoch=10
    ):
        r"""Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs

        :param optimizer: Optimizer used to optimize the policy model  
        :param epoch: The current epoch of the training process
        :param init_lr: Learning rate at which the training process starts off
        :param lr_decay_epoch: Number of epochs before the learning rate starts decaying
        :return optimizer: Model parameters with the updated learning rate 
        """
        lr = init_lr * (0.1**(epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    @staticmethod
    def generate_splits(num_episodes, 
        iteration_split
    ):
        r""" Splits the total number of episodes into a list over which the generation/training 
        cycles iterate over

        :param num_episodes: Integer. Total number of episodes for which the generation/training 
            cycles shall be carried out
        :param iteration_split: Integer. Number of episodes that will be used in one generation/training
            cycle
        :return t_iterations: List object. sum(t_iterations) = num_episodes
        """
        splits = num_episodes // iteration_split
        leftovers = num_episodes % iteration_split
        t_iterations = np.asarray([iteration_split for x in range(splits)], dtype = np.int32)
        
        if (leftovers != 0): t_iterations = np.append(t_iterations, leftovers)
        return t_iterations

    def train(self) -> None:
        r"""Trains the policy on the expert trajectories using the model utilities set in 
        the config.

        :return None
        """
        #initialize the common environment to use in this cycle
        dataset = make_dataset(self.config.TASK_CONFIG.DATASET.TYPE, 
                                    config = self.config.TASK_CONFIG.DATASET)
        env_class = get_env_class(self.config.ENV_NAME)
        self.env = env_class(config=self.config, dataset=dataset)

        logger.info("Initializing the expert agent")
        self.expert = Expert_Model(self.config, self.env)
        expert_config = self.config.IMITATION.EXPERT
        expert_dict = dict()

        #get the relevant varibles for initializing the training protocol
        num_episodes = expert_config.num_episodes
        split_dataset = expert_config.split_dataset
        iteration_split = expert_config.iteration_split

        #this uses the entire dataset    
        if(num_episodes == -1):
            num_episodes = len(dataset.episodes)

        #create an iterations list from the number of training episodes
        training_episodes = int(split_dataset*num_episodes)
        t_iterations = self.generate_splits(training_episodes, iteration_split)
        expert_dict['train'] = t_iterations

        #training dataset is being created, create folders if they don't exist
        image_path_train = expert_config.dataset_path.format(split = "train")
        self.expert.sanity_check(image_path_train)

        #create an iterations list if the split_dataset has been set
        validation_episodes = int(num_episodes - training_episodes)
        if(validation_episodes != 0):
            v_iterations = self.generate_splits(validation_episodes, iteration_split)
            expert_dict['val'] = v_iterations
            image_path_val = expert_config.dataset_path.format(split = "val")
            self.expert.sanity_check(image_path_val)

        #bc_config is the config node specific to the algorithm
        bc_config = self.config.IMITATION

        #loop over the train/val sets
        for phase, iterations_list in expert_dict.items():

            if(phase == 'train'): logger.info("Running the training-generation cycles")
            elif(phase == 'val'): logger.info("Generating a validation set to test the agent")

            for iterations in iterations_list:

                #generate the expert trajectories
                self.expert.generate_expert_trajectory(phase, iterations, expert_config.dataset_path, logger)

                #train the agent on these aggregated datasets
                if(phase == 'train'):
                    #set to True if there exists a trained model at the specified path
                    load_dict_exists = False

                    dataloader, size = self._get_dataloaders_size_dataset(bc_config, "train")

                    #create the checkpoint folder if it doesn't exist
                    if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
                        #gets called the first time the training-generation cycle runs
                        os.makedirs(self.config.CHECKPOINT_FOLDER)

                    #prior model file exists; called in other iterations
                    if os.path.isfile(os.path.join(self.config.CHECKPOINT_FOLDER, "policy_model.pth")):
                        load_dict = self.load_checkpoint(checkpoint_path=self.config.CHECKPOINT_FOLDER, 
                                                        checkpoint_index = -1,   
                                                        map_location="cpu"
                                                    )
                        load_dict_exists = True


                    #initialize the agent/load the agent
                    self._setup_agent(self.config)
                    if (load_dict_exists): self.agent.load_state_dict(load_dict["state_dict"]) 
                    self.agent.to(self.device)

                    if (not load_dict_exists): logger.info(
                        "agent number of parameters: {}".format(
                            sum(param.numel() for param in self.agent.parameters())
                        )
                    )

                    #model_utils
                    optimizer_ft = optim.Adam(self.agent.net.parameters(), 
                                            lr=bc_config.BC.learning_rate, 
                                            weight_decay= bc_config.BC.weight_decay,
                                            eps= bc_config.BC.eps
                                        )

                    criterion = nn.CrossEntropyLoss()
                    num_epochs = bc_config.BC.num_epochs
                    
                    logger.info(
                        "Number of epochs used for training the policy: {}".format(
                            num_epochs    
                        )
                    )
                    
                    #get the network
                    model = self.agent.net
                    if (not load_dict_exists): logger.info(f"The model architecture of the policy is: {model}")

                    with TensorboardWriter(
                        self.config.TENSORBOARD_DIR
                    ) as writer:

                        print ("Training the policy on the expert trajectories collected")
                        for epoch in tqdm(range(num_epochs)):
                            
                            #Schedule the learning rate every 10 epochs
                            optimizer_ft = self._exp_lr_scheduler(optimizer_ft, epoch, bc_config.BC.learning_rate)

                            running_loss = 0.0
                            running_corrects = 0

                            start_time = time.time()
                            
                            for sample in dataloader:
                                
                                expert_actions = sample['action']
                                expert_actions = expert_actions.to(self.device, non_blocking=True)
                                
                                # zero the parameter gradients
                                optimizer_ft.zero_grad()

                                outputs = model(sample, self.device)
                                _ , preds = torch.max(outputs, 1)

                                loss = criterion(outputs, expert_actions)

                                loss.backward()
                                optimizer_ft.step()

                                # statistics
                                running_loss += loss.item() * expert_actions.size(0)
                                running_corrects += torch.sum(preds == expert_actions.data)

                                del loss
                                del preds
                                del outputs
                                del expert_actions
                                del sample

                            #save checkpoints every 5 epochs
                            if (epoch!= 0 and (epoch+1)% bc_config.checkpoint_interval == 0):
                                self.save_checkpoint(f"checkpoint_{epoch+1}.pth")   
                                
                            #Epoch stats 
                            epoch_loss = running_loss / size
                            epoch_accuracy = running_corrects/ size

                            #Final epoch stats
                            logger.info(
                                    "Epoch_number: {}\tepoch_loss:{:.3f}\tepoch_accuracy:{}\ttime_taken:{:.3f}".format(
                                        epoch+1, 
                                        epoch_loss, 
                                        epoch_accuracy,
                                        time.time() - start_time
                                    )
                                )

                    #use the function to save the final model
                    del epoch_loss
                    del epoch_accuracy

                    self.save_checkpoint("policy_model.pth")

                #run validation if appropriate
                elif(phase == 'val'): 
                    self.validate()    

        return      
    
    def validate(self, 
        checkpoint_index:int = -1
    ) -> None:
        r"""Test the policy model on a held out test set to maintain consistency with the idea of 
        simplifying the general imitation learning problem to a supervised learning problem
        
        :param checkpoint_index: Integer. Default value = -1. Uses a policy defined by the model stored
            in that particular checkpoint. Using the default value uses a policy defined by the model 
            generated after the entire training process 
        :return None 
        """   
        #bc_config is the specific config for the algorithm
        bc_config = self.config.IMITATION

        if(checkpoint_index == -1):
            logger.info("Loading the final model to use as a part of the policy of the agent")
        
        else:
            logger.info("Loading the model saved at checkpoint {} as a part of the policy of the agent"
                .format(checkpoint_index)
            )

        gen_ckpt_dict = self.load_checkpoint(checkpoint_path=self.config.CHECKPOINT_FOLDER, 
                                            checkpoint_index = checkpoint_index,   
                                            map_location="cpu"
                                        )
               
        dataloader, size = self._get_dataloaders_size_dataset(bc_config, "val")
        criterion = nn.CrossEntropyLoss()

        #setup the model
        self._setup_agent(bc_config)
        self.agent.load_state_dict(gen_ckpt_dict["state_dict"])
        self.agent.to(self.device)

        model = self.agent.net

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR
        ) as writer:

            logger.info("Testing the trained policy on the validation set")

            running_loss = 0.0
            running_corrects = 0

            start_time = time.time()
            
            for sample in dataloader:
                
                expert_actions = sample['action']
                expert_actions = expert_actions.to(self.device)
                
                outputs = model(sample, self.device)
                loss = criterion(outputs, expert_actions)
                _ , preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * expert_actions.size(0)
                running_corrects += torch.sum(preds == expert_actions.data)

                del preds
                del outputs
                del expert_actions
                del sample
  
            #Epoch stats
            val_loss = running_loss/size
            val_accuracy = running_corrects/ size

            #print this out to the console
            logger.info("Validation stats here")
            logger.info(
                    "Val_loss:{}\tVal_accuracy:{}\ttime_taken:{:.3f}".format(
                        val_loss, 
                        val_accuracy,
                        time.time() - start_time
                    )
                )

        #use the function to save the final model
        del val_loss
        del val_accuracy

        return

    def eval_checkpoint(
        self,
        checkpoint_path: str,
        writer = TensorboardWriter,
        checkpoint_index: int = -1
    ) -> None:
        r"""Tests the algorithm in a free environment

        :param checkpoint_path:
        :param writer:
        :param checkpoint_index: The index used to load the model policy in a free environment. 
            Default value of -1 indicates loading the final trained model. 
        :return None
        """
        if(checkpoint_index == -1):
            logger.info("Loading the final model to use as a part of the policy of the agent")
        
        else:
            logger.info("Loading the model saved at checkpoint {} as a part of the policy of the agent"
                .format(checkpoint_index)
            )

        gen_ckpt_dict = self.load_checkpoint(checkpoint_path=self.config.CHECKPOINT_FOLDER, 
                                            checkpoint_index = checkpoint_index,   
                                            map_location="cpu"
                                        )
        
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(gen_ckpt_dict["config"])
        else:
            config = self.config.clone()

        bc_config = config.IMITATION
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        #setup the agent and load the state dict from the specified checkpoint
        self._setup_agent(bc_config)
        self.agent.load_state_dict(gen_ckpt_dict["state_dict"])
        self.agent.to(self.device)

        # get name of performance metric, e.g. "spl"
        metric_name = self.config.TASK_CONFIG.TASK.MEASUREMENTS[0]
        metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
        measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
        assert measure_type is not None, "invalid measurement type {}".format(
            metric_cfg.TYPE
        )
        self.metric_uuid = measure_type(
            sim=None, task=None, config=None
        )._get_uuid()

        #get observations and convert it into a tensor
        observations = self.envs.reset()
        batch = batch_obs(observations)

        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        stat_episodes = dict()

        rgb_frames = [[] 
            for i in range(self.config.NUM_PROCESSES)
        ]

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
            )


        with TensorboardWriter(
            self.config.TENSORBOARD_DIR
        ) as writer:

            while (
                len(stat_episodes) < self.config.TEST_EPISODE_COUNT
                and self.envs.num_envs > 0
            ):
                current_episodes = self.envs.current_episodes()

                with torch.no_grad():
                    (
                        actions
                    ) = self.agent.act(
                        batch,
                        self.device
                    )

                outputs = self.envs.step([a.item() for a in actions])

                observations, rewards, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]
                
                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )

                rewards = torch.tensor(
                    rewards, dtype=torch.float, device=self.device
                ).unsqueeze(1)
                current_episode_reward += rewards
                next_episodes = self.envs.current_episodes()
                
                #if a env needs to be paused
                envs_to_pause = []
                n_envs = self.envs.num_envs
                
                for i in range(n_envs):
                    if (
                        next_episodes[i].scene_id,
                        next_episodes[i].episode_id,
                    ) in stat_episodes:
                        envs_to_pause.append(i)

                    # episode ended
                    if not_done_masks[i].item() == 0:
                        episode_stat = dict()
                        episode_stat[self.metric_uuid] = infos[i][
                            self.metric_uuid
                        ]
                        episode_stat["success"] = int(
                            infos[i][self.metric_uuid] > 0
                        )
                        episode_stat["reward"] = current_episode_reward[i].item()
                        current_episode_reward[i] = 0
                        # use scene_id + episode_id as unique id for storing stats
                        stat_episodes[
                            (
                                current_episodes[i].scene_id,
                                current_episodes[i].episode_id,
                            )
                        ] = episode_stat

                        if len(self.config.VIDEO_OPTION) > 0:
                            generate_video(
                                video_option=self.config.VIDEO_OPTION,
                                video_dir=self.config.VIDEO_DIR,
                                images=rgb_frames[i],
                                episode_id=current_episodes[i].episode_id,
                                checkpoint_idx=checkpoint_index,
                                metric_name=self.metric_uuid,
                                metric_value=infos[i][self.metric_uuid],
                                tb_writer=writer,
                            )

                            rgb_frames[i] = []

                    # episode continues
                    if len(self.config.VIDEO_OPTION) > 0:
                        frame = observations_to_image(observations[i], infos[i])
                        rgb_frames[i].append(frame)

                (
                    self.envs,
                    not_done_masks,
                    current_episode_reward,
                    rgb_frames
                ) = self._pause_envs(
                    envs_to_pause,
                    self.envs,
                    not_done_masks,
                    current_episode_reward,
                    rgb_frames
                )

            aggregated_stat = dict()
            for stat_key in next(iter(stat_episodes.values())).keys():
                aggregated_stat[stat_key] = sum(
                    [v[stat_key] for v in stat_episodes.values()]
                )
            num_episodes = len(stat_episodes)

            episode_reward_mean = aggregated_stat["reward"] / num_episodes
            episode_metric_mean = aggregated_stat[self.metric_uuid] / num_episodes
            episode_success_mean = aggregated_stat["success"] / num_episodes

            logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
            logger.info(f"Average episode success: {episode_success_mean:.6f}")
            logger.info(
                f"Average episode {self.metric_uuid}: {episode_metric_mean:.6f}"
            )

            writer.add_scalars(
                "eval_reward",
                {"average reward": episode_reward_mean},
                checkpoint_index,
            )
            writer.add_scalars(
                f"eval_{self.metric_uuid}",
                {f"average {self.metric_uuid}": episode_metric_mean},
                checkpoint_index,
            )
            writer.add_scalars(
                "eval_success",
                {"average success": episode_success_mean},
                checkpoint_index,
            )

        self.envs.close()
        return

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        not_done_masks,
        current_episode_reward,
        rgb_frames,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            current_episode_reward,
            rgb_frames
        )

    def save_checkpoint(self, 
        file_name:str
    ) -> None:
        r"""Save the checkpoint with a specified name.

        :param file_name: String object. The file name to be used for the checkpoint
        :return None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

        return


    def load_checkpoint(self, 
        checkpoint_path:str, 
        checkpoint_index: int = -1, 
        *args, 
        **kwargs
    ) -> Dict:
        r"""Loads the specified checkpoint (saved during the training process) into memory

        :param checkpoint_path: String object. Path to the folder where the checkpoints are stored
        :param checkpoint_index: Integer. The index of the checkpoint that will be loaded 

        :return ckpt_dict: Dictionary object. A dictionary instance that has the state_dict and the 
        config file saved for that particular checkpoint
        """
        #checkpoint=-1 indicates that you want to load the final trained model
        if (checkpoint_index == -1):
            checkpoint_path = os.path.join(checkpoint_path, "policy_model.pth")

        else:
            checkpoint_path = os.path.join(checkpoint_path , "checkpoint_" + str(checkpoint_index) + ".pth")

        return torch.load(checkpoint_path, *args, **kwargs)