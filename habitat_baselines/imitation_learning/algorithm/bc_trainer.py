#!/usr/bin/env python3

import os
import time
from typing import Dict, List

import numpy as np
import time
from tqdm import tqdm

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
from habitat_baselines.imitation_learning.algorithm.policy import PointNavBaselinePolicy
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter


@baseline_registry.register_trainer(name="bc")
class BCTrainer(BaseTrainer):
    r"""
    Trainer class for Behavioral Cloning (BC)
    """

    supported_tasks = ["Nav-v0"]

    def __init__(self, config:Config):
        super().__init__()
        self.agent = None
        self.envs = None
        self.config = config
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if config is not None:
            logger.info(f"config: {config}")

    def _setup_agent(self, config):
        r"""Initializes an agent and a log file for the problem of PointGoal Navigation

        Args:
            config: A config file with the relevant params for initializing the agent

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)
        imitation_config = self.config.IMITATION

        self.agent = PointNavBaselinePolicy(
            observation_space = self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            hidden_size = imitation_config.BC.hidden_size
        )

    def _get_dataloaders(self, bc_config):
        r"""Creates dataloaders using an instance of the ExpertDataset

        Args:
            bc_config: config node with the relevant params for behavioral cloning

        Returns:
            dataloaders: dataloaders to load data from the train anf validation sets 
        """
        #append the npz extensions
        train_path = bc_config.EXPERT.train_path + ".npz"
        val_path = bc_config.EXPERT.val_path + ".npz"

        #create instances of the ExpertDataset
        image_datasets = {x: ExpertDataset(y) for (x,y) in [('train',train_path), ('val',val_path)]}
        
        dataloaders = {x: DataLoader(image_datasets[x], 
                        batch_size=bc_config.BC.batch_size, 
                        shuffle=bc_config.BC.shuffle, 
                        num_workers=bc_config.BC.num_workers
                    ) for x in ['train', 'val']
            }
                
        return dataloaders

    def _get_size_dataset(self, bc_config=None):
        r"""Gets the size of the datasets for calculating epoch stats

        Args:
            bc_config: config node with the relevant params for behavioral cloning

        Returns:
            Dict(str, int): contains the size of the train and val folders
        """
        
        train_path = bc_config.EXPERT.train_path + ".npz"
        val_path = bc_config.EXPERT.val_path + ".npz"

        return {
            "train_size": len(np.load(train_path)['rgb']),
            "val_size": len(np.load(val_path)['rgb'])
        }

    @staticmethod
    def _exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=10):
        r"""Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs

        Args:
            optimizer: Optimizer used to optimize the policy model  
            epoch: The current epoch of the training process
            init_lr: Learning rate at which the training process starts off
            lr_decay_epoch: Number of epochs before the learning rate starts decaying

        Returns:
            optimizer: Model parameters with the updated learning rate 
        """
        
        lr = init_lr * (0.1**(epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    #taken from the BaseRLTrainer class
    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)
        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = "val"

        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

    def train(self) -> None:
        r"""Trains the policy on the expert trajectories using the model utilities set in 
        the config.

        Args:
            None

        Returns:
            None 
        """

        self.envs = construct_envs(self.config, 
                                    get_env_class(self.config.ENV_NAME)
                                )

        #bc_config is the specific config for the algorithm
        bc_config = self.config.IMITATION
        
        dataloaders = self._get_dataloaders(bc_config)
        sizes = self._get_size_dataset(bc_config)

        #create the checkpoint folder if it doesn't exist
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        #initialize the agent
        self._setup_agent(self.config)
        self.agent.to(self.device)

        logger.info(
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
        logger.info(f"The model architecture of the policy is: {model}")

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR
        ) as writer:

            print ("Training the policy on the expert trajectory")
            for epoch in tqdm(range(num_epochs)):
                
                #Schedule the learning rate every 10 epochs
                optimizer_ft = self._exp_lr_scheduler(optimizer_ft, epoch, bc_config.BC.learning_rate)

                running_loss = 0.0
                running_corrects = 0

                start_time = time.time()
                
                for sample in dataloaders['train']:
                    
                    expert_actions = sample['action']
                    expert_actions = expert_actions.to(self.device)
                    
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
                if (epoch!= 0 and (epoch+1)%5 == 0):
                    self.save_checkpoint(f"checkpoint_{epoch+1}.pth")   
                    
                #Epoch stats 
                epoch_loss = running_loss / sizes['train_size']
                epoch_accuracy = running_corrects/ sizes['train_size']

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
        self.envs.close()  
        
    def validate(self, checkpoint_index:int = -1) -> None:
        r"""Test the policy model on a held out test set to maintain consistency with the idea of 
        simplifying the general imitation learning problem to a supervised learning problem
        
        Args:
            None

        Returns:
            None
        """
        #bc_config is the specific config for the algorithm
        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))

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
               
        dataloaders = self._get_dataloaders(bc_config)
        sizes = self._get_size_dataset(bc_config)
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
            
            for sample in dataloaders['val']:
                
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
            val_loss = running_loss/sizes['val_size'] 
            val_accuracy = running_corrects/ sizes['val_size']

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
        self.envs.close()

    def eval(self, checkpoint_index:int = -1) -> None:
        r"""Tests the algorithm in a free environment

        Args:
            checkpoint_index: The index used to load the model policy in a free environment. Default value of
            -1 indicates loading the final trained model. 

        Returns: None
        
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

    def save_checkpoint(self, file_name:str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )


    def load_checkpoint(self, checkpoint_path:str, checkpoint_index: int = -1, *args, **kwargs) -> Dict:
        r"""Loads the specified checkpoint (saved during the training process) into memory

        Args: 
            checkpoint_path: Path to the folder where the checkpoints are stored
            checkpoint_index: The index 

        Returns:
            A Dictionary instance that has the state_dict and config file saved for that particular
            checkpoint
        """
        #checkpoint=-1 indicates that you want to load the final trained model
        if (checkpoint_index == -1):
            checkpoint_path = os.path.join(checkpoint_path, "policy_model.pth")

        else:
            checkpoint_path = os.path.join(checkpoint_path , "checkpoint_" + str(checkpoint_index) + ".pth")
        
        return torch.load(checkpoint_path, *args, **kwargs)
