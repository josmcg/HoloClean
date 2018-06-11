from errordetector import ErrorDetection
from holoclean.learning.prediction_layer import LogReg
from torch import nn as nn, optim as optim
from holoclean.errordetection.augmentor import Augmentor

class HoloDetect(ErrorDetection):
    def __init__(self,train_seed, learning_params, dataset, featurizers):
        """
        Initialize the HoloDetect module
        :param train_seed: A set of training data in the form of a
            data frame with cols [index, attribute, value, label]
        :param learning_params: learning related parameters in a dictionary
            #TODO create a default set of learning parameters
        :param dataset: the dataset as a dataframe
        :param featurizers: A list of featurizer classes
        """
        self.featurizers = featurizers
        self.train_seed = train_seed
        self.dataset = dataset
        self.learning_params = learning_params
        self.clean = None
        self.dirty = None
        self.augmentor = Augmentor(train_seed)
        self.find_errors()

    def find_errors(self):
        """
        this method supervises learning and eventually populates the
        dirty and clean variables
        :return: None
        """
        model = LogReg(self.featurizers, 1)
        criterion = nn.CrossEntropyLoss()
        params = nn.ParameterList(model.parameters())
        for featurizer in self.featurizers:
            params.append(featurizer.parameters())
        optimizer = optim.SGD(params, lr=self.learning_params["lr"])
        # bootstrap the model with known labeled data
        for i in range(self.learning_params["bootstrap_epochs"]):
            for batch in self.augmentor.batches():
                optimizer.zero_grad()
                # TODO call some function to prepare featurizers
                # TODO the LogReg Class does not apply the nonlinearity
                outputs = model.forward()
                loss = criterion(outputs, batch.labels)
                loss.backward()
                optimizer.step()
                # TODO add learning related printing for verbose mode
        # here we add examples at each step if we have a high enough certainty
        for i in range(self.learning_params["secondary_epochs"]):
            pass

        # predict for the whole dataset
        outputs = model.forward()
        for (idx,output) in enumerate(outputs):
            if output > self.learning_params["error_threshold"]:
                self.add_to_set("dirty", idx)
            else:
                self.add_to_set("clean",idx)

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

