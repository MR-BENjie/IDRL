# IDRL
We develop IDRL, a novel multi-agent reinforcement learning framework consisting of identification and policy modules that learns the policy with identification capability. Here, we release the code about IDRL implementation in the Red-10 game, Red-10 game evrionment and visualize tool.
## IDRL Implementation in the Red-10 Game
In the directory "IDRL_in_Red_10", we release the IDRL implementation in the Red-10 Game, including train and evaluate code. The Framework of IDRL is visualized in the figure below.

<img width="500" src="https://github.com/MR-BENjie/IDRL/raw/main/over_all_framework.jpg"/>

The core concept of the IDRL framework is to transform a setting with ambiguous agent identities into one with less ambiguous identities. That is, agents are empowered to intuitively infer the identity of a cooperating agent and then act upon the assumption. In the IDRL framework, we use the identification module to identify others identities first; then, the policy module, which is pretrained with appropriate action sets, generates operational rule sets. Recall that the identification module comprises a relation network which generates a confidence level vector, and a danger network which generates a risk ratio. The confidence level and risk ratio are then combined to select a corresponding policy in the policy module. Then, the agent acts upon the selected policy.

### installation
The training code is designed for GPUs, thus, you need to first install CUDA.

First, Make sure you have python 3.6+ installed. Install dependencies.
```
cd IDRL_in_Red_10
pip3 install -r requirements.txt
```

Then, the Red_10 game environment should be installed. The detail information is in the directory "IDRL_in_Red_10/IDRL/env/rlcard_red10"

### Training
First, we should train policy sets for policy module, we use parallel training method for training. 

To use GPU for training policy module, run
```
python3 train.py
```
This will train several policy sets of policy module on one GPU. To train on multiple GPUs. Use the following arguments.
```
`--gpu_devices`: what gpu devices are visible
`--num_actor_devices`: how many of the GPU deveices will be used for simulation, i.e., self-play
`--num_actors`: how many actor processes will be used for each device
`--training_device`: which device will be used for training 
```

For example, if we have 4 GPUs, where we want to use the first 3 GPUs to have 15 actors each for simulating and the 4th GPU for training, we can run the following command:
```
python3 train.py --gpu_devices 0,1,2,3 --num_actor_devices 3 --num_actors 15 --training_device 3
```

For more customized configuration of training, see the following optional arguments:
```
--xpid XPID           Experiment id
--save_interval SAVE_INTERVAL
                      Time interval (in minutes) at which to save the model
--objective {adp,wp}  Use ADP or WP as reward (default: ADP)
--actor_device_cpu    Use CPU as actor device
--gpu_devices GPU_DEVICES
                      Which GPUs to be used for training
--num_actor_devices NUM_ACTOR_DEVICES
                      The number of devices used for simulation
--num_actors NUM_ACTORS
                      The number of actors for each simulation device
--training_device TRAINING_DEVICE
                      The index of the GPU used for training models. `cpu`
                	  means using cpu
--load_model          Load an existing model
--disable_checkpoint  Disable saving checkpoint
--savedir SAVEDIR     Root dir where experiment data will be saved
--total_frames TOTAL_FRAMES
                      Total environment frames to train for
--exp_epsilon EXP_EPSILON
                      The probability for exploration
--batch_size BATCH_SIZE
                      Learner batch size
--unroll_length UNROLL_LENGTH
                      The unroll length (time dimension)
--num_buffers NUM_BUFFERS
                      Number of shared-memory buffers
--num_threads NUM_THREADS
                      Number learner threads
--max_grad_norm MAX_GRAD_NORM
                      Max norm of gradients
--learning_rate LEARNING_RATE
                      Learning rate
--alpha ALPHA         RMSProp smoothing constant
--momentum MOMENTUM   RMSProp momentum
--epsilon EPSILON     RMSProp epsilon
```
The policy module parameter file will be saved in the "IDRL_in_Red_10/douzero_checkpoints"

Then, we get four policy sets for four cooperation--competition patterns: 1100,1010,1000 and 0000. And for each pattern,
we get four policies for four players: landlord, landlrod_up, landlord_front, landlord_down.
 
Before training relation network and danger network, we should place the policy parameter file in specified location,
for example, landlord_up policy in 1000 pattern should be place in location below.  
```
'IDRL_in_Red_10/douzero_checkpoints/douzero_1000/model/landlord_up_weights.ckpt'
```

Then, the relation network and danger network can be trained, and we also use parallel training. Run train_RD.py, the
detail usage just like the "train.py" above. 
### evaluate policy module 
#### step one：generate evaluation data
```
python3 generate_eval_data.py
```

Here, we assigned deck for four players.

Some important hyperparameters are list below.
```
 `--output`: pickle file saved path
 `--num_games`: number of Red-10 game rounds did. default number: 10000
```
#### step two: self-play
```
python3 evaluate.py
```
Some important hyperparameters are list below.
```
`--landlord`: agent 0's pre-trained policy parameter path 
`--landlord_up`: agent 3's pre-trained policy parameter path
`--landlord_down`: agent 1's pre-trained policy parameter path
`--landlord_front`: agent 2's pre-trained policy parameter path
`--eval_data`: the pickle file contains evaluation data
`--num_workers`: the number of subprocesses used 
`--gpu_device`: GPU device used. 
```

### evaluate relation and danger networks in Red_10
We use the evaluate_red_10.py to evaluate the performance of the relation network and danger network. We use it to visualize the 
confidence level and risk ratio transformation in different rounds.

Before evaluation, the policy parameter file should be placed in specified path noticed above, and the parameter file of ralation
network and danger network should be organized as 'IDRL_in_Red_10/R_D_checkpoints/relation_weights.ckpt' and 'IDRL_in_Red_10/R_D_checkpoints/dangerous_weights.ckpt' 

## Red-10 environment with evaluation and visualization tool

In Red_10 directory, we provide the code of the Red-10 game environment for reinforcement learning, and evaluation and visualization tool for Red-10 game.

The Red-10 game environment is in the subdirectory "Red_10/rlcard"

### Red_10 game environment
<img width="500" src="https://dczha.com/files/rlcard/logo.jpg" alt="Logo" />
RLCard is a toolkit for Reinforcement Learning (RL) in card games. We use the RLCard to create the Red-10 game environment.

#### installation 
Make sure you have installed **Python 3.6+** and **pip**.

Installed rlcard using pip.
```
pip3 install rlcard
```
The default installation will only include the card environments. To use PyTorch implementation of the training algorithms, run
```
pip3 install rlcard[torch]
```

Then installed with
```
cd Red_10
pip3 install -e .
pip3 install -e .[torch]
```

### Evaluation and visualization tool for Red-10 game.
We create and release the evaluation and visualization tool for Red-10 game in subdirectory rlcard-showdown, which help 
understand the performance of the agents. The usage of tool is list in the subdirectory. 