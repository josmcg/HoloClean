from torch import nn
import torch
from holoclean.global_variables import  GlobalVariables
import pandas as pd
from tqdm import tqdm


class CoOccurED(nn.Module):
    def __init__(self,session):
        super(CoOccurED, self).__init__()
        self.session = session
        self.dataengine = session.holo_env.dataengine
        self.df = self.dataengine.get_table_to_dataframe("Init", session.dataset)
        self.cols = self.df.schema.names
        self.count = len(self.cols) - 1

    def forward(self, example):
        index, attr, value = example
        # just query the original dataset for every other attribute
        # see how many times the original pair of (attr, other) show up
        ret = torch.zeros(len(self.cols) - 1)
        index, attr, value = example
        same_val = self.df.filter("{} = '{}'".format(attr, value))
        # recover original row
        og_row = same_val.filter(GlobalVariables.index_name + " = " + str(index)).collect()[0]
        for (idx, col) in enumerate(self.cols):
            if col == GlobalVariables.index_name:
                continue
            attr_val = og_row[col]
            cooccur = same_val.filter("{} = '{}'".format(col, attr_val))
            ret[idx] = cooccur.count()
        return ret



