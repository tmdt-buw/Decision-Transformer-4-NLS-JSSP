# Dispatching Decision Transformer #
This project contains the code for the Dispatching
Decision Transformer (D-DT) integrated in to the
_schlably_ project.

## Setup
   ```sh
   pip install -r requirements.txt
   ```
Also see _schlably_ readme.

## Basic usage
To try the model use _compare_models.py_. It runs the one of the DT models and its origin agent on the dataset defined
in _6x6_agent_mppo_config.yaml_.

### Create new train Dataset
To create a new training dataset for the Decision Transformer run the dataset_creation.py
It will solve the dataset defined in the _6x6_agent_mppo_config.yaml_ and extract datapoints for DT training.

### Train new Model
In order to train a new model run _main.py_ with specified train_dataset.

## Documentation
### D-DT Source Code
All files related to the D-DT are in the path: _src/agents/disptaching-decision-transformer_.
All following described directory's and files are accordingly in the _src/agents/dispatching-decision-transformer_ directory.

### minGPT ### 
 Contains the code of the original [Decision Transformer](https://github.com/kzl/decision-transformer/ "Named link title") which is based on [minGPT](https://github.com/karpathy/minGPT).
#### dt_model.py #### 
Contains the minGPT pytorch Model adjusted for the D-DT
forward path of the model expects _sequence_ of states, actions, targets rtgs,timestep, action_masks
Note that minGPT refers to the context_length as _block_size_.

**Class: dt_model.GPTConfig(vocab_size, block_size, kwargs)**
vocab_size -> amount of actions in action space
block_size -> context_length  
Furthermore defines via kwargs ->  
n_layer -> number of transformer blocks to use
n_head -> number of self-attention heads  
n_embd -> embedding size  
max_timestep -> Max length of RL environment  
observation_size


#### dt_trainer.py ####
This file defines the minGPT training framework

Contains: 

**TrainerConfig**:

Set various parameters like learning rate, batch size, and optimization settings. It includes settings for learning rate decay.
**Trainer**: 

The main class responsible for training the model. Implements a training loop with epoch and batch processing, and includes methods for saving checkpoints and testing the model. Logsprogress to Weights & Biases (wandb).

#### utils.py ####
Implements the funcitonality to pass the sequences to model 
and converts logits returned from the model to the predicted action.


#### StateActionReturnDataSet.py ####
Class designed for use with PyTorch's Dataset interface.
The state action return dataset is necessary for training. On intialization data in the
format created from _create_data_set_ function is necessary.

__getitem()__ 
returns  for a given index the sequence of the next _context_length_ _observations_, _actions_, rtgs, _timesteps_, _action_masks_
and time steps. 

#### DispatchingDecisionTransformer.py ###
Implements usage of a trained DT model for inference.

### Experiments ##
The  experiments directory contains the compare_models.py and the var_rtgs.py

**compare_models.py** 

The implements functionality to the mppo origin-agent and a
specified D-DT model in the same environment with the same test instances and returns 
their achieved mean makespan.


**vary_rtgs.py**

Implements functionality to run specified D-DT models to solve the same test instances
with various return-to-go values.
Also usable for creating the Boxplots for hamming and start time distance for rtg=1

_Datasets_ (Only available on DVD because of github size restriction)
Directory contains the Datasets with the diffrent observation strategies and diffrent action 
selection of the origin agent (Stochastic or deterministic).


## Trained models
The trained models used for the experiments in the thesis are in _lib/trained-models_
Each problem size has the NLSDT and the NLSDT_100 model.


