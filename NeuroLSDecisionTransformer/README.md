# NeuroLS Decision Transformer Documentation
## Setup
**Note:** For me using Windows the NeuroLS environment is only working when using Windows Subsytem for Linux  (WSL)  

Install the requirements as conda environment
```sh
conda env create -f requirements.yml
```
## Basic Usage
Run Decision Transformer 15x15 model on Taillard Benchmark   
(For other problem change respective entries in command):

```sh
python run_benchmark.py -r run_nls_jssp.py -d  data/JSSP/benchmark/TA -g 15x15 -p jssp -m nls -e eval_jssp --args "env=jssp15x15_unf" -n 200 -x  15x15/NLSDT_100/brisk-lake-1290.pt -a True --rtg_factor 1.0 
```
Note that in the _eval_jssp.yaml_ the NLS model for the respective problem size must be uncommented.
Further flags are shown in _run_benchmark.py_


Run NLS 15x15 model on Taillard Benchmark
```sh
python run_benchmark.py -r run_nls_jssp.py -d  data/JSSP/benchmark/TA -g 15x15 -p jssp -m nls -e eval_jssp --args "env=jssp15x15_unf" -n 200
```

#### run_benchmark.py
Handles to run benchmark or test dataset with the NLS or NLSDT (depending on if -a flag is True)  
**Important:** _tester_cfg.test_dataset_size= in line 112 must be 1 for Taillard benchmark and 100 (or other testdataset size) otherwise.
Watch out to add a space after the number in the string.

### Setting up a new dataset to train Neuro-LS Decision Transformer
<ol>
    <li> Create dataset of jssp instances with <i>create_jssp_val_test_sets.ipynb</i> for respective problem size 
    </li>
<li> Set <i>env</i>: in <i>jssp_config.yaml</i> to respective problem size</li>
<li> Define values in eval_jssp.yaml 
    Make sure to set:
    <ul>
        <li>run_type: must allways be test (We do not want to train the original NLS Model)</li>
        <li>(Uncomment repsective <i>checkpoint:</i> </li>
        <li>test_dataset_size: Size of the dataset that is used for NLS-DT data creation.</li>
        <li>test_batch_size: Must be 1 for NLS-DT</li>
        <li>Operator mode: <b>SET</b>(for NLS_A action space), <b>SELECT_LS(for NLS_AN)</b>, <b>SELECT_LS+</b>(for NLS_ANP)</li>
        <li>num_steps: Number of iterations to run the local search per instance</li>
        <li>data_file_path: Path to instance file created with <i>create_jssp_val_test_sets.ipynb</i> notebook</li>
    </ul>
</li>
<li><b>run_nls_jssp.py</b>: This will run the NeuroLS model over the test_instance file defined in <i>eval_jssp.yaml</i>.
</li>
<li>Process the created data points with <i>create_dt_dataset.ipynb</i> to calculate returns-to-go and convert format of dataset</li>
</ol>

### Train NeuroLS Decision Transformer ###
In order to train a new model run _lib/main.py_ with specified previously created train dataset.
**Important:** The trainer (either dt_trainer or dt_cassandra_trainer) needs to be adjusted to the problem size and the validation dataset.
The run_benchmark.py also needs to be adjusted for the respective training data set size. Also allways the right NLS model needs to be activated in the _eval_jssp.yaml.
#### Traindata
The used train data is only available on the DVD-ROM appended to the thesis(Path: data/dt_dataset) because it exceeds the github file size limit.


## Documentation
### minGPT ### 
 Contains the code of the original [Decision Transformer](https://github.com/kzl/decision-transformer/ "Named link title") which is based on [minGPT](https://github.com/karpathy/minGPT).
#### dt_model.py #### 
Contains the minGPT pytorch Model adjusted for the D-DT
forward path of the model expects _sequence_ of states, actions, targets rtgs,timestep, action_masks
Note that minGPT refers to the context_length as _block_size_.

**dt_model.GPTConfig(vocab_size, block_size, kwargs)**  
vocab_size -> amount of actions in action space
block_size -> context_length  
Furthermore defines via kwargs ->  
n_layer -> number of transformer blocks to use
n_head -> number of self-attention heads  
n_embd -> embedding size  
max_timestep -> Max length of RL environment  
observation_size

#### dt_trainer.py ####
This file define the minGPT training framework  
Contains: 

**TrainerConfig**:  
Sets various parameters like learning rate, batch size, and optimization settings. It includes settings for learning rate decay.

**Trainer**:  
The main class responsible for training the model. Implements a training loop with epoch and batch processing, and includes methods for saving checkpoints and testing the model. Logsprogress to Weights & Biases (wandb).

#### dt_cassandra_trainer.py ####
Equivalent to dt_trainer but for parallel multi gpu training

#### utils.py ####
Implements the functionality to pass the sequences to the minGPT model 
and converts logits returned from the model to the predicted action.


### lib/utils ###
#### tianshou_utils.py ####
<u>_TestCollector.collect()_</u>
Implements the inference of the NLS model and the NLSDT model. In case of DT usage the aggregated state is extracted and processed to the DecisionTransfomer object
Extracts datapoints for NLSDT when solving instances and saves data_points.pt to output directory

#### StateActionReturnDataSet.py ####
Class designed for use with PyTorch's Dataset interface.
The StateActionReturndDataSet is necessary for training. On initialization data in the
format created from _create_data_set.py_ is necessary.  
__getitem()__ 
returns  for a given index the sequence of the next _context_length_ _observations_, _actions_, rtgs, _timesteps_ 
and time steps. 

**Important**: The vocab_size defined in _StateActionReturnDataSet_ must be selected according to the operator mode.  
SET = 2; SELECT_LS = 8; SELECT_LS+ = 10


### lib/ ###
#### DecisionTransformer.py ###
Defines the DecisionTransformer class. The class includes methods for calculating return-to-go, updating rewards and states, and sampling actions based on the aggregated environment 

#### Experiments ####
To reproduce the exact same experiments presented in the thesis without needing to run the whole vary_rtg() process the zip file _experiment_output/nls.zip_
and _experiment_output/nlsdt.zip_ need to be unzipped.
The folders contain numpy files for each solved instance named by a unique name per instance in order to compare the instances that are
solved by the NLS and NLSDT.

Experiments for rtg variation are done in _lib/experiments/vary_rtgs.py_. For comparing nls and nlsdt for only one rtg value 
the _lib/experiments/compare_models_and_plot.ipynb_ notebook can be used.


### Important NLS files with additional changes made for NLSDT ### 
**<u>_lib/env/jssp_env.py_:</u>** 

Gym environment to solve Job Shop Scheduling Problems based on Local Search.
Here it was necessary to add the following to the observation:
<ul>
<li>'instance_lower_bound'</li>
<li>'instance_hash'</li>
<li>'machine_sequence'</li>
<li>'starting_times'</li>
</ul>
Also the calc_lower_bound() method is implemented in this environment class


**<u>_lib/scheduling/jssp_graph.py_:</u>**  
Manages the whole disjunctive graph of an instance. Further needed
to get machine sequence and starting times of instance.


**<u>_lib/networks/model.py_</u>**: Model that creates the embedding. The aggregated State that is used as observation for
the Decision Transformer is created here.
