from holoclean.global_variables import GlobalVariables
from torch.utils.data import Dataset
from pyspark.sql.window import *
import pandas as pd
import pandas.io.sql as sqlio
from sklearn.model_selection import train_test_split
import torch

class Augmentor:
    def __init__(self, ground_truth_path, session, augmentation_rules=None):
        """
        Initialize an augmentation dataset handler
        :param ground_truth: a subset of cleaned data, flattened
        :param augmentation_rules: a set of functions
            which take a an example [index, attribute, value, label] and a dataset (frame) and
            produce more cells with new labels
        :param session: a holoclean session object
        """
        self.augmentation_rules = augmentation_rules
        self.session = session
        self.conn = session.holo_env.dataengine.db_backend[1]
        self.ground_truth_path = ground_truth_path
        self.labeled_set = self._labeled_table()

    def add_example(self, example):
        """
        add an example to the labeled set, and apply augmentation rules to it
        :param example:
        :return:
        """
        self.labeled_set.append(example)
        for f in self.augmentation_rules:
            self.labeled_set.append(f(example, self.dataset))

    def _labeled_table(self):
        """
        flattens the ground truth into a labeled set
        :return:
        """
        flattened_init = self._flatten_init(self.session)
        ground_truth = self.read_ground_truth(self.session)
        joined = flattened_init.merge(ground_truth, on=["rv_index", "rv_attr"], suffixes=["","_truth"])
        joined["error"] = joined.apply(lambda row: not(row["attr_val"] == row["attr_val_truth"]), axis = 1)
        return joined

    def get(self):
        return PySparkDataset(self.labeled_set)

    def split(self, frac):
        """
        split the known values into a train-test split
        :param frac: the fraction of examples to be used as training data
        :return: the train and test frames as (train, test)
        """
        train, test = train_test_split(self.labeled_set, test_size=1-frac)
        return PySparkDataset(train)

    def _flatten_init(self, session):
        """
            given a holoclean session,generate a flattened init table
            """
        cols = session.dataset.get_schema("Init")
        cols.remove(GlobalVariables.index_name)
        # decide is we need to create a new table
        exists = False
        try:
            query_check = "SELECT * FROM " + session.dataset.table_specific_name("Init_flat")
            session.holo_env.dataengine.query(query_check)
            exists = True
        except:
            pass

        if not exists:
            query_create = "CREATE TABLE \
                            " + session.dataset.table_specific_name('Init_flat') \
                           + "( rv_index TEXT, \
                            rv_attr TEXT, attr_val TEXT);"
            session.holo_env.dataengine.query(query_create)
            # now with the table certainly created and empty, we populate it
            for col in cols:
                # get all cols
                query_get = "(SELECT DISTINCT t1." + GlobalVariables.index_name + \
                            " as rv_index " + \
                            ", \'" + col + "\' as rv_attr , t1." + \
                            col + " as attr_val" + \
                            "  FROM " + session.dataset.table_specific_name("Init") + \
                            " as t1)"
                query_insert = "INSERT INTO " + session.dataset.table_specific_name("Init_flat") + \
                               " " + query_get
                session.holo_env.dataengine.query(query_insert)
        else:
            pass
        return sqlio.read_sql_query("SELECT * FROM {}".format(session.dataset.table_specific_name("Init_flat")), self.conn)

    def read_ground_truth(self, session):
        """
        turns the path given at initialization into a dataframe
        :return: a pyspark dataframe
        """

        query_create = "CREATE TABLE " + session.dataset.table_specific_name("ground_truth") + \
                       "(rv_index TEXT, rv_attr TEXT, attr_val TEXT)"
        session.holo_env.dataengine.query(query_create)
        # parse the lines in the file
        with open(self.ground_truth_path) as handle:
            # use map when you implement this robustly
            for line in [line.strip() for line in handle][1:]:
                items = line.split(",")
                query_insert = "INSERT INTO {} VALUES ('{}', '{}', '{}')".format(
                    session.dataset.table_specific_name("ground_truth"),
                    items[0],
                    items[1],
                    items[2]
                )
                session.holo_env.dataengine.query(query_insert)

        return sqlio.read_sql_query("SELECT * FROM {}".format(session.dataset.table_specific_name("ground_truth")), self.conn)

    def test(self):
        return PySparkDataset(self.labeled_set)


class PySparkDataset(Dataset):
    def __init__(self, df):
        super(Dataset, self).__init__()
        self.df = df

    def __getitem__(self, index):
        # handle the different index methods
        returned_df = self.df.iloc[index]
        row = returned_df.to_dict()
        return row["rv_index"], row["rv_attr"], row["attr_val"], row["error"].astype(int)

    def __len__(self):
        return self.df.shape[0]

    def create_table(self, df, id, session):
        dataengine = session.holo_env.dataengine
        table_id = session.dataset.table_specific_name(id)
        dataengine.dataframe_to_table(table_id, df)
        return table_id



