import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import Flatten


class ModifiedCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        hidden_size: Output size of the final linear layer in the CNN subsection of the model
        output_size: The size of the action space of the agent
        goal_sensor_uuid: The sensor that determines how close the agent is to it's goal
    """

    def __init__(self, observation_space, hidden_size, output_size, goal_sensor_uuid):
        super().__init__()
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0
            
        if goal_sensor_uuid in observation_space.spaces:
            self._n_input_pg = observation_space.spaces[goal_sensor_uuid].shape[0]
            self.compass_available = True
        else:
            self._n_input_pg = 0
            self.compass_available = False
        
        #print ("The value of input_rgb", self._n_input_rgb) -> 3 channels that is being held by this variable
        #print ("The value of input_depth", self._n_input_depth) -> 1 channels 
        
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )
            
        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_rgb + self._n_input_depth,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            #  nn.ReLU(True),
            Flatten(),
            nn.Linear(32* cnn_dims[0] * cnn_dims[1], hidden_size),
            nn.ReLU(True)

        )
        
        #used if the agent is equipped with a gps+compass sensor
        self.compass_mlp = nn.Sequential(
            nn.Linear(hidden_size + 2, output_size),
            #nn.ReLU(True)
            #nn.Linear(2 * cnn_dims[0] * cnn_dims[1], output_size)
        )
        
        #used otherwise
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            #nn.ReLU(True)
            #nn.Linear(2 * cnn_dims[0] * cnn_dims[1], output_size)
        )

        self.layer_init(self.compass_available)

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self, compass_available):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        
        if (self.compass_available):
            for layer in self.compass_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("relu")
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
        
        else:
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("relu")
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations, device):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations.float()
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)
            
        if self._n_input_pg > 0:
            pg_observations = observations['pointgoal_with_gps_compass']
            pg_observations = pg_observations.to(device)

        cnn_input = torch.cat(cnn_input, dim=1)
        cnn_input = cnn_input.to(device)

        embed = self.cnn(cnn_input)
        
        del cnn_input
        del rgb_observations
        del depth_observations

        if self.compass_available:
            concat_embed = torch.cat((embed, pg_observations), 1)
            return self.compass_mlp(concat_embed)
        else:
            return self.mlp(embed)