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
        self.test_N = None
        self.test_L = None
        self.tensor_train = None
        self.tensor_test = None

        # These values must be overridden in subclass
        self.offset = 0  # offset on the feature_id_map
        """
        The type of the featurizer will determine the way we tie the weights 
        in the prediction module. See options above
        """
        self.count = 0

    def forward(self, clean=1):
        """
        Forward step of the featurizer
        Creates the tensor for this specific feature. Should not be
        overridden
        """
        if clean:
            #training

            if self.tensor_train is None:
                self.tensor_train = self.create_tensor(clean, self.N, self.L)
            else:
                if self.update_flag:
                    # if the weights are updated we need to create again the tensor
                    self.tensor_train =  self.create_tensor(clean, self.N, self.L)
            return self.tensor_train
        else:
            #testing
            if self.test_N is None or self.test_L is None:
                raise EnvironmentError("test dimensions not set")
            if self.tensor_test is None:
                self.tensor_test = self.create_tensor(clean, self.test_N, self.test_L)
            else:
                if self.update_flag:
                    # if the weights are updated we need to create again the tensor
                    self.tensor_test = self.create_tensor(clean, self.test_N, self.test_L)
            return self.tensor_test

    def set_test_dimensions(self, N, L):
        self.test_N = N
        self.test_L = L

    @abstractmethod
    def create_tensor(self, clean=1, N=None, L=None):
        """
        This method creates the tensor for the feature
        """
        tensor = None
        return tensor





