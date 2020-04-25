Imitation Learning
==============================

## Behavioral Cloning (BC)

### Setup:
1. Use the commands listed below
```
git clone --branch imitation-learning git@github.com:wannabeOG/habitat-api.git
cd habitat-api
pip install -r requirements.txt
python setup.py develop --all # install habitat and habitat_baselines
```
2. Follow rest of the instructions listed [here][16] to install habitat-sim  
3. Download the [test scenes data][18] and extract ```data``` folder in zip to ```habitat-api/data/``` where ```habitat-api/``` is the github repository folder.

### Dependencies: 
PyTorch 1.0, for installing refer to [pytorch.org][1]

### Directory Structure: 
```
imitation_learning
├── algorithm
│   ├── bc_trainer.py
│   ├── dataset.py
│   ├── expert.py
│   ├── policy.py
├── models
│   ├── cnn.py
```

### Files:
1. ``algorithm/bc_trainer.py``: A trainer file that implements the ideas of behavioral cloning.
1. ``algorithm/dataset.py``: Enables loading the data from Expert dataset in batches. 
1. ``algorithm/expert.py``: Expert agent which is used to train an agent through behavioral cloning.
1. ``algorithm/policy.py``: A general implementation of the policy of an agent.
1. ``models/cnn.py``: A modified implementation of the [SimpleCNN model][2].


### [Config File][12]:

```IMITATION.EXPERT``` is a record of the attributes required to generate the expert trajectories

1. ***mode***: Can take values ```greedy``` or ```geodesic```. Refers to the mode which the shortest path follower(expert) adopts to pick the best action when interacting with the environment
1. ***num_episodes***: Number of episodes that the expert will interact with the environment with for
1. ***split_dataset***: A floating point value between 0 and 1 which decides the number of episodes that will be used to create the training set
1. ***iteration_split***: Number of episodes that will be used in one generation/training cycle
1. ***path***: Path where the dataset of expert trajectories (train/val) is stored at. 
1. ***log_file***: Name of the log file

```IMITATION.BC``` is a record of the attributes required to train a policy using Behavioral Cloning

**Optimizer used**: [torch.optim.Adam][14]  
**Loss Criterion**: [nn.CrossEntropyLoss][15]

Optimizer attributes (Check the docs linked above for more information):
1. ***weight_decay***: weight decay (L2 penalty) 
1. ***eps***: term added to the denominator to improve numerical stability
1. ***learning_rate***: Initial learning rate. Learning rate is scheduled to decay every 10 epochs

Dataloader attributes:

1. ***batch_size***: Batch size used whilst loading the data from the dataset
1. ***num_workers***: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process

Check [here][17] for more information

1. ***num_epochs***: Number of epochs you wish to run the training process for
1. ***hidden_size***: The [output size][13] of the final linear layer used in the CNN model  
1. ***checkpoint_interval***: A checkpoint will be stored at the appropriate location every "checkpoint_interval

### Model Architecture Used

This is an instance created by using the default values of the config
```
ModifiedCNN(
  (cnn): Sequential(
    (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    (1): ReLU(inplace)
    (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    (3): ReLU(inplace)
    (4): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
    (5): Flatten()
    (6): Linear(in_features=25088, out_features=512, bias=True)
    (7): ReLU(inplace)
  )
  (compass_mlp): Sequential(
    (0): Linear(in_features=514, out_features=4, bias=True)
  )
  (mlp): Sequential(
    (0): Linear(in_features=512, out_features=4, bias=True)
  )
)
```

```compass_mlp``` is a linear layer that is used if the agent is equipped with a (gps+compass) sensor  
```mlp``` is a linear layer used in the case if the agent does not have the above facility

**Generate Expert Trajectories and Train and Validate the Policy on them**

```bash
python -u habitat_baselines/run_test.py --exp-config habitat_baselines/config/pointnav/bc_pointnav.yaml --run-type train
```

**Test**
```bash
python -u habitat_baselines/run_test.py --exp-config habitat_baselines/config/pointnav/bc_pointnav.yaml --run-type eval --checkpoint-number ()
```
Checkpoint number is used to load the model from a specific checkpoint created during the training routine of the policy. If this field is left empty, the final checkpoint is used 

### Reported Results
Will be updated shortly

### Assumptions and Current Limitations:
1. Aesthetic issues: While using the [habitat logger][11] during the training phase of the policy, the progress bar breaks since the imported logger is initialized with a stream handler, so by default the logger prints the output to the console which breaks the progress bar. One possible solution would be to import HabitatLogger and initialize it with a file_name. This would however not print anything to the console (STDOUT) and just store a record in the log file. At present though, I have gone along by using the standard ```logger.info``` to make a record so this will result in broken progress bars whilst training the policy model on expert trajectories.     

### References
1. [Imitation Learning Tutorial][20]  
1. [Repository for SplitNet][21]
1. [Stable Baslines Repository][5]

[1]:https://pytorch.org/
[2]:https://github.com/wannabeOG/habitat-api/blob/imitation-learning/habitat_baselines/rl/models/simple_cnn.py
[3]:https://github.com/wannabeOG/habitat-api/blob/b9ef56db96ff7e3551534d530296794139ef9e24/habitat/tasks/nav/shortest_path_follower.py#L28
[4]:https://github.com/facebookresearch/habitat-api/blob/f5e29c69e5ba35704ca8b4e0c5e43dca89191845/examples/shortest_path_follower_example.py#L71
[5]:https://github.com/hill-a/stable-baselines
[6]:https://github.com/wannabeOG/habitat-api/blob/b9ef56db96ff7e3551534d530296794139ef9e24/habitat_baselines/imitation_learning/algorithm/bc_trainer.py#L33
[7]:https://github.com/wannabeOG/habitat-api/blob/b9ef56db96ff7e3551534d530296794139ef9e24/habitat_baselines/common/base_trainer.py#L18
[8]:https://github.com/wannabeOG/habitat-api/blob/b9ef56db96ff7e3551534d530296794139ef9e24/habitat_baselines/imitation_learning/algorithm/bc_trainer.py#L137
[9]:https://github.com/wannabeOG/habitat-api/blob/b9ef56db96ff7e3551534d530296794139ef9e24/habitat_baselines/imitation_learning/algorithm/bc_trainer.py#L596
[10]:https://github.com/wannabeOG/habitat-api/blob/b9ef56db96ff7e3551534d530296794139ef9e24/habitat_baselines/common/base_trainer.py#L39
[11]:https://github.com/wannabeOG/habitat-api/blob/b9ef56db96ff7e3551534d530296794139ef9e24/habitat/core/logging.py#L24
[12]:https://github.com/wannabeOG/habitat-api/blob/imitation-learning/habitat_baselines/config/pointnav/bc_pointnav.yaml
[13]:https://github.com/wannabeOG/habitat-api/blob/b9ef56db96ff7e3551534d530296794139ef9e24/habitat_baselines/imitation_learning/models/cnn.py#L91
[14]:https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
[15]:https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
[16]:https://github.com/facebookresearch/habitat-api#installation
[17]:https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
[18]:http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip
[19]:https://github.com/wannabeOG/habitat-api/blob/b9ef56db96ff7e3551534d530296794139ef9e24/habitat_baselines/imitation_learning/algorithm/dataset.py#L10
[20]:https://sites.google.com/view/icml2018-imitation-learning/
[21]:https://github.com/facebookresearch/splitnet/tree/master/supervised_learning
