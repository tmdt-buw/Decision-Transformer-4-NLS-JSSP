import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import numpy as np

# constants
SHARED_FEATURE_SIZE = 18


class EncoderModule(nn.Module):

    def __init__(self, embedding_dim, attention_heads: int = 2, hidden_dim: int = 32):
        super(EncoderModule, self).__init__()
        # structure
        self.multiHeadAttention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=attention_heads)
        self.norm = nn.LayerNorm(embedding_dim)
        # TODO implement variable nn structure here according to policy kwargs
        self.feed_forward = nn. Sequential(
                                            nn.Linear(embedding_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, inputs):

        # self attention
        attention, _ = self.multiHeadAttention(inputs, inputs, inputs)
                                               #key_padding_mask=self.get_key_padding_mask(inputs))
        # add and norm
        x = inputs + attention
        x = self.norm(x)
        # feed forward
        linear_out = self.feed_forward(x)
        # add and norm
        x = x + linear_out
        x = self.norm(x)
        return x

    @staticmethod
    def get_key_padding_mask(obs: torch.Tensor):
        shape = obs.shape
        mask = None
        # single case (env step)
        if len(shape) == 2:

            mask = torch.zeros(obs.shape[0])

            for i, task in enumerate(obs):
                if torch.sum(task) == 0:
                    mask[i] = 1

        # batched case (train)
        if len(shape) == 3:
            mask = torch.zeros(obs.shape[0], obs.shape[1])

            for i, single_obs in enumerate(obs):
                for j, task in enumerate(single_obs):
                    if torch.sum(task) == 0:
                        mask[i][j] = 1

            # resize to match PyTorch attention modul prerequisites
            # TODO does this reshape what we want?
            # TODO breakpoint -> look if mask is still equal in Dot attention module
            mask = mask.reshape(mask.shape[1], mask.shape[0])

        return mask


class CombinedPolicyEncoder(nn.Module):
    """
    Attention and NN policy Combined
    """

    # TODO use all arguments from kwargs in function head
    def __init__(self, action_dim, obs_dim, num_tasks, learning_rate, hidden_dim=32, **kwargs):
        super(CombinedPolicyEncoder, self).__init__()

        # TODO use for 1D obs
        self.shared_feature_dim = SHARED_FEATURE_SIZE
        task_feature_dim = obs_dim[0] - self.shared_feature_dim
        self.input_dim = num_tasks
        assert task_feature_dim % num_tasks == 0, "Number of features must be divisible by number of tasks "
        self.emb_dim = int(task_feature_dim / num_tasks)

        # TODO hard coded to check if observation runs with ppo_masked and this module
        # self.shared_feature_dim = SHARED_FEATURE_SIZE
        # self.input_dim = 36
        # self.emb_dim = 10


        # TODO assertions and assignments for observation with shape array!!!
        # assert len(obs_dim) > 1, \
        #     "To use a CombinedModule policy you have to choose an observation with two dimensions"
        #
        # # unwrap observation dim
        # task_feature_dims = obs_dim[0]
        # self.shared_feature_dim = obs_dim[1]
        #
        # assert len(task_feature_dims) > 1, \
        #     "To use a ValueEncoder policy you have to choose an observation with two dimensions"
        #
        # # unwrap dims for the attention module
        # self.input_dim = task_feature_dims[0]  # number of tasks in the input -> first dimension of the obs
        # self.emb_dim = task_feature_dims[1]  # number_of features (per task_vector) -> second dimension of the obs

        # structure
        structure = []

        # Attention: stack the requested number of encoder modules
        for _ in range(kwargs['num_modules']):
            structure.append(EncoderModule(self.emb_dim, kwargs['num_heads']))

        self.encoder = nn.Sequential(*structure)

        # MLP
        self.linear = nn.Sequential(
            nn.Linear(self.shared_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.shared_feature_dim),
            nn.ReLU()
        )

        # Output mapping
        self.output_layer = nn.Linear(self.input_dim * self.emb_dim + self.shared_feature_dim, action_dim)
        self.softmax = nn.LogSoftmax(dim=1)

        #
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs_batch, action_mask=None):

        batch_case = True
        if not len(inputs_batch.shape) > 1:
            batch_case = False

        if not batch_case:
            inputs_batch = [inputs_batch]

        for i, inputs in enumerate(inputs_batch):

            # reshape 1D input into 1D shared feature and 2D task_features for attention module
            task_features, shared_features = torch.split(inputs, [self.input_dim*self.emb_dim, self.shared_feature_dim])
            task_features = task_features.reshape(self.input_dim, self.emb_dim)

            # pass task features through attention
            x_1 = self.encoder(task_features)
            x_1 = x_1.view(self.input_dim * self.emb_dim)

            # pass shared features through MLP
            x_2 = self.linear(shared_features)

            # concatenate both lanes and do output activation
            x = torch.concat((x_1, x_2))

            # concat results of each batch
            if batch_case:
                if i > 0:
                    x = torch.concat((last_x, x))
                last_x = x

        if batch_case:
            # reshape flattened batches back
            x = x.reshape((256, self.input_dim * self.emb_dim + self.shared_feature_dim))

        x = self.output_layer(x)
        x = x.view(-1, 6)
        softmax_out = self.softmax(x)
        # mask probabilities if action_mask is not None (for env.reset)
        if action_mask is not None:
            action_mask.to(self.device)
            softmax_out = torch.where(action_mask, softmax_out, torch.tensor(-1e+8).to(self.device))
        dist = Categorical(logits=softmax_out)

        return dist


class CombinedValueEncoder(nn.Module):
    """
    Attention and NN policy Combined
    """

    # TODO use all arguments from kwargs in function head
    def __init__(self, obs_dim, num_tasks, learning_rate, hidden_dim=32, **kwargs):
        super(CombinedValueEncoder, self).__init__()

        # TODO use for 1D obs
        self.shared_feature_dim = SHARED_FEATURE_SIZE
        task_feature_dim = obs_dim[0] - self.shared_feature_dim
        self.input_dim = num_tasks
        assert task_feature_dim % num_tasks == 0, "Number of features must be divisible by number of tasks "
        self.emb_dim = int(task_feature_dim / num_tasks)

        # TODO hard coded to check if observation runs with ppo_masked and this module
        # self.shared_feature_dim = SHARED_FEATURE_SIZE
        # self.input_dim = 36
        # self.emb_dim = 10

        # structure
        structure = []

        # Attention: stack the requested number of encoder modules
        for _ in range(kwargs['num_modules']):
            structure.append(EncoderModule(self.emb_dim, kwargs['num_heads']))

        self.encoder = nn.Sequential(*structure)

        # MLP
        self.linear = nn.Sequential(
            nn.Linear(self.shared_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.shared_feature_dim),
            nn.ReLU()
        )

        # Output mapping
        self.output_layer = nn.Linear(self.input_dim * self.emb_dim + self.shared_feature_dim, 1)

        #
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs_batch):

        batch_case = True
        if not len(inputs_batch.shape) > 1:
            batch_case = False

        if not batch_case:
            inputs_batch = [inputs_batch]

        for i, inputs in enumerate(inputs_batch):

            # reshape 1D input into 1D shared feature and 2D task_features for attention module
            task_features, shared_features = torch.split(inputs, [self.input_dim * self.emb_dim, self.shared_feature_dim])
            task_features = task_features.reshape(self.input_dim, self.emb_dim)

            x_1 = self.encoder(task_features)
            x_1 = x_1.view(self.input_dim * self.emb_dim)

            x_2 = self.linear(shared_features)

            x = torch.concat((x_1, x_2))

            # concat results of each batch
            if batch_case:
                if i > 0:
                    x = torch.concat((last_x, x))
                last_x = x

        if batch_case:
            # reshape flattened batches back
            x = x.reshape((256, self.input_dim * self.emb_dim + self.shared_feature_dim))

        x = self.output_layer(x)

        return x

class PolicyEncoder(nn.Module):
    def __init__(self, action_dim, obs_dim, num_tasks, learning_rate, **kwargs):
        super(PolicyEncoder, self).__init__()

        self.input_dim = obs_dim[0]  # number of tasks in the input -> first dimension of the obs
        assert len(obs_dim) > 1, \
            "To use a ValueEncoder policy you have to choose an observation with two dimensions"
        self.emb_dim = obs_dim[1]  # number_of features (per task_vector) -> second dimension of the obs

        # structure
        structure = []
        # stack the requested number of encoder modules
        for _ in range(kwargs['num_modules']):
            structure.append(EncoderModule(self.emb_dim, kwargs['num_heads']))

        self.encoder = nn.Sequential(*structure)

        self.linear = nn.Linear(self.input_dim*self.emb_dim, action_dim)
        self.softmax = nn.LogSoftmax(dim=1)

        #
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs, action_mask=None):

        x = self.encoder(inputs)
        x = x.view(-1, self.input_dim*self.emb_dim)
        x = self.linear(x)
        softmax_out = self.softmax(x)
        # mask probabilities if action_mask is not None (for env.reset)
        if action_mask is not None:
            action_mask.to(self.device)
            softmax_out = torch.where(action_mask, softmax_out, torch.tensor(-1e+8).to(self.device))
        dist = Categorical(logits=softmax_out)
        return dist


class ValueEncoder(nn.Module):
    def __init__(self, obs_dim, num_tasks, learning_rate, **kwargs):
        """

        :param action_dim:
        :param obs_dim:
        :param embedding_dim: Determines 1 dimension of the input matrix. Can be equal to action if one obs_vector for each job
        :param attention_heads:
        """
        super(ValueEncoder, self).__init__()

        self.input_dim = obs_dim[0]  # number of tasks in the input -> first dimension of the obs
        assert len(obs_dim) > 1, \
            "To use a ValueEncoder policy you have to choose an observation with two dimensions"
        self.emb_dim = obs_dim[1]  # number_of features (per task_vector) -> second dimension of the obs

        # structure
        structure = []

        # stack the requested number of encoder modules
        for _ in range(kwargs['num_modules']):
            structure.append(EncoderModule(self.emb_dim, kwargs['num_heads']))

        self.encoder = nn.Sequential(*structure)

        # create linear output layer to map encoder representation to one value
        self.linear = nn.Linear(self.input_dim * self.emb_dim, 1)

        #
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs):

        x = self.encoder(inputs)
        x = x.view(-1, self.input_dim*self.emb_dim)
        x = self.linear(x)

        return x
