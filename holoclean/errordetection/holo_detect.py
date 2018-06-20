from errordetector import ErrorDetection
import torch
from torch import nn as nn, optim as optim
from holoclean.errordetection.augmentor import Augmentor

class HoloDetect(ErrorDetection):
    def __init__(self, train_seed, session, learning_params=None, featurizers=None):
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
        self.test = None
        self.train = None
        #self.find_errors()

    def find_errors(self):
        """
        this method supervises learning and eventually populates the
        dirty and clean variables
        :return: None
        """
        self.train, self.test = self.get_split(frac=.1)
        params = nn.ParameterList()
        feature_count = 0
        for featurizer in self.featurizers():
            params.extend(featurizer.parameters())
            feature_count += featurizer.count
        linear_layer = nn.Linear(feature_count,1)
        params.extend(linear_layer.parameters())
        sigmoid = nn.Sigmoid()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(params, lr=0.1)
        labels = self.train.select("error").collect()
        labels = [(lambda cell: 1 if cell else -1)(cell) for cell in labels]
        y = torch.tensor(labels)
        for epoch in range(self.learning_params["epochs"]):
            optimizer.zero_grad()
            representation = self.featurizers[0].forward()
            for featurizer in self.featurizers[0:]:
                sub_tensor = featurizer.forward()
                representation = torch.cat((representation,sub_tensor),1)
            # we must pare down  the featurization to only have the rows we care about
            scores = linear_layer(representation)
            preds = sigmoid(scores)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()


    def get_all(self):
        return self.augmentor.get()

    def get_split(self, frac):
        return self.augmentor.split(frac)

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

