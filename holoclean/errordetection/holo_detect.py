from errordetector import ErrorDetection
import torch
from torch import nn as nn, optim as optim
from holoclean.errordetection.augmentor import Augmentor
from torch.utils.data import DataLoader
from tqdm import tqdm

default_learning_params = {"epochs": 2, "lr": 0.1, "threshold": 0.88}


class HoloDetect(ErrorDetection):
    def __init__(self, train_seed, session, featurizers= None ,learning_params=default_learning_params):
        """
        Initialize the HoloDetect module
        :param train_seed: path to clean data
        :param session: a HoloClean session object
        :param learning_params: learning related parameters in a dictionary
        :param featurizers: A list of featurizer classes
        """
        self.featurizers = featurizers
        self.session = session
        self.train_seed = train_seed
        self.learning_params = learning_params
        self.clean = None
        self.dirty = None
        self.augmentor = Augmentor(train_seed,session)
        self.train = None

    def find_errors(self):
        """
        this method supervises learning and eventually populates the
        dirty and clean variables
        :return: None
        """
        self.train= self.get_split(frac=.1)
        num_pos = self.train.df.filter("error = True").count()
        print("train has {} errors".format(num_pos))
        print("train has {} clean examples".format(self.train.df.count()- num_pos))
        params = nn.ParameterList()
        feature_count = 0
        for featurizer in self.featurizers:
            params.extend(featurizer.parameters())
            feature_count += featurizer.count
        linear_layer = nn.Linear(feature_count, 1)
        params.extend(linear_layer.parameters())
        sigmoid = nn.Sigmoid()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(params, lr=self.learning_params["lr"])
        train_loader = DataLoader(self.train)
        for epoch in range(self.learning_params["epochs"]):
            tot = float(len(self.train))
            agg = 0.0
            for (idx, example) in tqdm(enumerate(train_loader)):
                data = example[:3]
                data = [item[0] for item in data]
                label = example[3].float()
                optimizer.zero_grad()
                representation = self.featurizers[0].forward(data)
                for featurizer in self.featurizers[1:]:
                    sub_tensor = featurizer.forward(data)
                    representation = torch.cat((representation, sub_tensor), 0)
                scores = linear_layer(representation)
                preds = sigmoid(scores)
                loss = criterion(preds, label)
                agg += loss.item()
                loss.backward()
                optimizer.step()
            summary_str = "epoch {}: loss: {}".format(epoch, agg/float(tot))
            print(summary_str)
        print("finished training model")
        # prediction phase
        all_examples = self.get_all()
        preds = []
        true_labels = []
        data_loader = DataLoader(all_examples)
        for (idx, example) in enumerate(data_loader):
            data = example[:3]
            data = [item[0] for item in data]
            label = example[3].float()
            true_labels.append(label)
            optimizer.zero_grad()
            representation = self.featurizers[0].forward(data)
            for featurizer in self.featurizers[1:]:
                sub_tensor = featurizer.forward(data)
                representation = torch.cat((representation, sub_tensor), 0)
            scores = linear_layer(representation)
            if bool(scores.ge(.88)):
                preds.append(torch.ones(1))
            else:
                preds.append(torch.zeros(1))
        # evaluate
        tot = len(preds)
        corr = 0.0
        err_caught = 0.0
        total_err = num_pos
        for (idx, pred) in enumerate(preds):
            truth = true_labels[idx]
            if truth == pred:
                corr += 1
                if truth == 1:
                    err_caught += 1
        print("precision is {}".format(corr/tot))
        print("found {} of {} total errors".format(err_caught, total_err))



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

