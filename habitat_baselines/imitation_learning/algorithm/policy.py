#!/usr/bin/env python3

import abc

import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import Flatten
from habitat_baselines.imitation_learning.models.cnn import ModifiedCNN

class Policy(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        device
    ):

        #deterministic= False does it really exist in the case of BC?
        actions_distributions = self.net(observations,device)
        _, preds = torch.max(actions_distributions, 1)
        return preds
        
class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size,
    ):
        super().__init__(
            ModifiedCNN(
                observation_space=observation_space,
                hidden_size=hidden_size,
                output_size=action_space.n,
                goal_sensor_uuid=goal_sensor_uuid,
            )
        )
