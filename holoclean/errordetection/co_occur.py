from torch import nn
import torch
from holoclean.global_variables import  GlobalVariables
import pandas as pd
import pandas.io.sql as sqlio
import numpy
from holoclean.utils.pruning import Pruning

class CoOccurED(nn.Module):
    def __init__(self,session):
        super(CoOccurED, self).__init__()
        self.session = session
        self.dataengine = session.holo_env.dataengine
        self.df = self.dataengine.get_table_to_dataframe("Init", session.dataset)
        self.conn = session.holo_env.dataengine.db_backend[1]
        self.cols = self.df.schema.names
        self.count = len(self.cols) - 1
        self.df = sqlio.read_sql_query("SELECT * FROM {}"\
                                       .format(session.dataset.table_specific_name("Init")), self.conn)





    def forward(self, example):
        # just query the original dataset for every other attribute
        # see how many times the original pair of (attr, other) show up
        index, attr, value = example
        og_row = self.df[(self.df[attr.lower()] == value) & (self.df[GlobalVariables.index_name] == int(index))].to_dict('records')[0]
        value = value.decode('utf-8')
        value = unicode(value)
        query_str_pre = u"SELECT COALESCE(count(*),0) FROM {0} as t1  WHERE t1.{1} = '{2}' "\
            .format(unicode(self.session.dataset.table_specific_name("Init")),
                    unicode(attr),
                    unicode(value),
                    unicode(GlobalVariables.index_name))



        # recover original row
        queries = []
        for (idx, col) in enumerate(self.cols):
            if col == GlobalVariables.index_name:
                continue
            attr_val = og_row[col]
            try:
                attr_val = attr_val.decode('utf-8')
                attr_val = unicode(value)
            except Exception as e:
                print(attr_val)
            cooccur_str = u" AND t1.{} = '{}'".format(col, attr_val)
            query_str = u"({} {})".format(query_str_pre, cooccur_str)
            queries.append(query_str)
        final_query = u' UNION ALL'.join(queries)
        cooccur = sqlio.read_sql_query(final_query, self.conn)
        num = cooccur.values
        return torch.from_numpy(num).squeeze()



