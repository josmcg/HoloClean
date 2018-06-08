import torch.nn as nn
from abc import ABCMeta, abstractmethod

class Featurizer(nn.Module):

    __metaclass__ = ABCMeta

    def __init__(self, N, L,  update_flag=False):
        """
        Creates a pytorch module which will be a featurizer for HoloClean
        :param n : number of random variables
        :param l: number of classes
        :param update_flag: True if the values in tensor of the featurizer
        need be updated

        """
        super(Featurizer, self).__init__()
        self.update_flag = update_flag
        self.N = N
        self.L = L
        self.tensor = None

        # These values must be overridden in subclass
        self.offset = 0  # offset on the feature_id_map
        """
        The type of the featurizer will determine the way we tie the weights
        0. No tieing
        1. Tieing the weights in the domain
        """
        self.type = None
        self.count = 0
        self.id = "Base"

    def forward(self):
        """
        Forward step of the featurizer
        Creates the tensor for this specific feature
        """
        if self.tensor is None:
            self.create_tensor()
        else:
            if self.update_flag:
                #if the weights are updated we need to create again the tensor
                self.create_tensor()

        return self.tensor

    @abstractmethod
    def create_tensor(self):
        """
        This method creates the tensor for the feature
        """
        return self.tensor





