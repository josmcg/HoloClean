from torch import nn
import torch
from holoclean.global_variables import  GlobalVariables
import pandas as pd
from tqdm import tqdm
import time
import numpy

class CoOccurED(nn.Module):
    def __init__(self,session):
        super(CoOccurED, self).__init__()
        self.session = session
        self.dataengine = session.holo_env.dataengine
        self.df = self.dataengine.get_table_to_dataframe("Init", session.dataset)
        self.cols = self.df.schema.names
        self.count = len(self.cols) - 1

    def forward(self, example):
        # just query the original dataset for every other attribute
        # see how many times the original pair of (attr, other) show up
        ret = torch.zeros(len(self.cols) - 1)
        index, attr, value = example
        query_str_pre = u"SELECT COALESCE(count(*),0) FROM {0} as t1  WHERE t1.{1} = '{2}' "\
            .format(self.session.dataset.table_specific_name("Init"),
                    attr,
                    value,
                    GlobalVariables.index_name)



        # recover original row
        og_query_str_pre = u"SELECT * from {} WHERE {} = '{}'".format(self.session.dataset.table_specific_name("Init"),
                                                            attr,
                                                            value)
        og_row_str = u" AND {} ={}".format(GlobalVariables.index_name, index)
        og_row = self.dataengine.query(og_query_str_pre+og_row_str, 1).collect()[0]
        queries = []
        for (idx, col) in enumerate(self.cols):
            if col == GlobalVariables.index_name:
                continue
            attr_val = og_row[col]
            cooccur_str = u" AND t1.{} = '{}'".format(col, attr_val)
            query_str = u"({} {})".format(query_str_pre, cooccur_str)
            queries.append(query_str)
        final_query = u' UNION ALL'.join(queries)
        cooccur = self.dataengine.query(final_query, 1)
        num = numpy.array(cooccur.collect())
        return torch.from_numpy(num).squeeze()



