#
from abc import abstractmethod
from typing import Union, Optional
import torch.nn as nn

from lib.networks.formats import Obs, GraphObs, Emb


class BaseEncoder(nn.Module):
    """Abstract encoder model."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 **kwargs):
        """

        Args:
            input_dim: dimension of features
            output_dim: embedding dimension of output
            hidden_dim: dimension of hidden layers
        """
        super(BaseEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def create_layers(self, **kwargs):
        """Create the specific model layers."""
        raise NotImplementedError

    def reset_parameters(self):
        """Reset the trainable model parameters."""
        pass

    def forward(self, obs: Union[Obs, GraphObs], emb: Emb = Emb(), **kwargs) -> Emb:
        """
        Model specific implementation of forward pass.

        Args:
            obs: batched observation tuple
            emb: RPEmb tuple

        Returns:
            emb: tuple of created embeddings

        """
        raise NotImplementedError


