from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np


class StateActionReturnDataset(Dataset):
    def __init__(self, data: List[dict], context_length):
        self.data = data
        self.block_size = context_length
        self.vocab_size = 10 #Depends on operator mode
        self.problem_size = 200 #Number of iterations

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, datapoint_index):
        instance_step_number = self.data[datapoint_index].get("step_number")

        if instance_step_number + self.block_size > (self.problem_size):
            datapoint_index -= (instance_step_number + self.block_size - self.problem_size) #Ensures the context_length will not include values of the next instance when step number is to close to the end of an instance

        observation_sequence_array = np.array([instance.get("observation")
                                               for instance in self.data[datapoint_index:datapoint_index + self.block_size]])
        action_sequence_array = np.array([instance.get("action")
                                          for instance in self.data[datapoint_index:datapoint_index + self.block_size]])
        target_sequence_array = np.array([instance.get("action")
                                          for instance in self.data[datapoint_index:datapoint_index + self.block_size]])
        return_to_go_sequence_array = np.array([instance.get("returns_to_go")
                                                for instance in self.data[datapoint_index:datapoint_index + self.block_size]])

        states = torch.tensor(observation_sequence_array, dtype=torch.float32).reshape(self.block_size, -1) #(block_size, 128)
        actions = torch.tensor(action_sequence_array, dtype=torch.long).unsqueeze(1) # (block_size, 1)
        returns_to_go = torch.tensor(return_to_go_sequence_array, dtype=torch.float64).unsqueeze(1) # (block_size, 1) #NOTE for neuro ls we need float rtgs
        timesteps = torch.tensor([self.data[datapoint_index].get("step_number")], dtype=torch.int64).unsqueeze(1) #(block_size, 1)
        targets = torch.tensor(target_sequence_array, dtype=torch.long).unsqueeze(1) # (block_size, 1)

        return states, actions, returns_to_go, timesteps, targets#, action_maskstargets
