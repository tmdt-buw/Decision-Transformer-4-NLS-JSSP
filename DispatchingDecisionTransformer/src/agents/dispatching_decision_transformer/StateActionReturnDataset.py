from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np


class StateActionReturnDataset(Dataset):
    def __init__(self, data: List[dict], context_length):
        self.data = data
        self.context_length = context_length
        self.vocab_size = 6 #Amount of actions
        self.problem_size = 6*6 #actions*machines

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, datapoint_index):
        instance_step_number = self.data[datapoint_index].get("step_number")

        if instance_step_number + self.context_length > (self.problem_size):
            datapoint_index -= (instance_step_number + self.context_length - self.problem_size) # Ensures the
            # context_length will not include values of the next instance when step number is to close to the end of
            # an instance

        #Retrieve
        observation_sequence_array = np.array([instance.get("observation")
                                               for instance in self.data[datapoint_index:datapoint_index + self.context_length]])
        action_sequence_array = np.array([instance.get("action")
                                          for instance in self.data[datapoint_index:datapoint_index + self.context_length]])
        return_to_go_sequence_array = np.array([instance.get("returns_to_go")
                                                for instance in self.data[datapoint_index:datapoint_index + self.context_length]])
        action_mask_sequence_array = np.array([instance.get("action_mask")
                                               for instance in self.data[datapoint_index:datapoint_index + self.context_length]])

        states = torch.tensor(observation_sequence_array, dtype=torch.float32).reshape(self.context_length, -1) #(block_size, 57)
        returns_to_go = torch.tensor(return_to_go_sequence_array, dtype=torch.long).unsqueeze(1) # (block_size, 1)
        timesteps = torch.tensor([self.data[datapoint_index].get("step_number")], dtype=torch.int64).unsqueeze(1) #(block_size, 1)
        actions = torch.tensor(action_sequence_array, dtype=torch.long).unsqueeze(1) # (block_size, 1)
        action_masks = torch.tensor(action_mask_sequence_array, dtype=torch.bool).unsqueeze(1)

        return states, actions, returns_to_go, timesteps, action_masks
