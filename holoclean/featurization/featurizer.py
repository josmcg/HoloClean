import torch.nn as nn
from abc import ABCMeta, abstractmethod


class Featurizer(nn.Module):
    NO_TIE = 0
    TIE_ALL = 1
    TIE_PER_FEATURE = 2  # Not yet implemented

    __metaclass__ = ABCMeta

    def __init__(self, N, L,  update_flag=False):
        """
        Creates a pytorch module which will be a featurizer for HoloClean
        :param N : number of random variables
        :param L: number of classes
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
        in the prediction module. See options above
        """
        self.type = None
        self.count = 0
        self.id = "Base"

    def forward(self):
        """
        Forward step of the featurizer
        Creates the tensor for this specific feature. Should not be
        overridden
        """
        if self.tensor is None:
            self.create_tensor()
        else:
            if self.update_flag:
                # if the weights are updated we need to create again the tensor
                self.create_tensor()

        return self.tensor

    @abstractmethod
    def create_tensor(self):
        """
        This method creates the tensor for the feature
        """
        return self.tensor





