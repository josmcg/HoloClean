from errordetector import ErrorDetection
from holoclean.learning.prediction_layer import LogReg
from torch import nn as nn, optim as optim
from holoclean.errordetection.augmentor import Augmentor

class HoloDetect(ErrorDetection):
    def __init__(self,train_seed,session, learning_params=None, featurizers=None):
        """
        Initialize the HoloDetect module
        :param train_seed: path to clean data
        :param session: a HoloClean session object
        :param learning_params: learning related parameters in a dictionary
            #TODO create a default set of learning parameters
        :param featurizers: A list of featurizer classes
        """
        self.featurizers = featurizers
        self.session = session
        self.train_seed = train_seed
        self.learning_params = learning_params
        self.clean = None
        self.dirty = None
        self.augmentor = Augmentor(train_seed,session)
        #self.find_errors()

    def find_errors(self):
        """
        this method supervises learning and eventually populates the
        dirty and clean variables
        :return: None
        """
        training_data = self.get_training_data()

    def get_training_data(self):
        return self.augmentor.get()

    def add_to_set(self,setname, idx):
        """
        adds a cell to the specified set
        :return:
        """
        pass

    def get_noisy_cells(self):
        return self.dirty

    def get_clean_cells(self):
        return self.clean

