from holoclean.global_variables import GlobalVariables
from holoclean.utils.parser_interface import DenialConstraint
import time
import torch
from torch import nn
import pandas as pd
import pandas.io.sql as sqlio
import numpy

__metaclass__ = type


class SqlDCErrorDetection(nn.Module):
    """
    This class is a subclass of ErrorDetection class and
    will returns don't know cells and clean cells based on the
    denial constraints
    """

    def __init__(self, session):
        """
        This constructor converts all denial constraints
        to the form of SQL constraints
        :param session: Holoclean session
        """
        super(SqlDCErrorDetection, self).\
            __init__()
        self.dataengine = session.holo_env.dataengine
        self.dataset = session.dataset
        self.spark_session = session.holo_env.spark_session
        self.holo_obj = session.holo_env
        self.session = session
        self.index = GlobalVariables.index_name
        self.dc_parser = session.parser
        self.operationsarr = DenialConstraint.operationsArr
        self.noisy_cells = None
        self.dc_objects = session.dc_objects
        self.Denial_constraints = session.Denial_constraints
        self.dfs = None
        self.count = len(self.Denial_constraints)
        self.features = None
        self.table_ids = None
        self.cache = None
        self.conn = session.holo_env.dataengine.db_backend[1]
        self.get_noisy_cells()

    # Internals Methods
    @staticmethod
    def _is_symmetric(dc_name):
        """
        Identifying symmetric denial constraint

        :param dc_name: denial constraint
        :return: boolean value
        """
        result = True
        non_sym_ops = ['<=', '>=', '<', '>']
        for op in non_sym_ops:
            if op in dc_name:
                result = False
        return result

    def _get_noisy_cells_for_dc(self, dc_name, table_id):
        """
        Returns a dataframe that consist of index of noisy cells index and
        attribute

        :param dc_name: denial constraint
        :return: spark_dataframe
        """

        if self.holo_obj.verbose:
            self.holo_obj.logger.info(
                'Denial Constraint Queries For ' + dc_name)
        t3 = time.time()
        dc_object = self.dc_objects[dc_name]
        temp_table = "tmp" + self.dataset.dataset_id

        # Create Query for temp table
        query = "CREATE TABLE " + temp_table +\
                " AS SELECT "
        for tuple_name in dc_object.tuple_names:
            query += tuple_name + "." + self.index + " as " + \
                     tuple_name + "_ind,"

        query = query[:-1]
        query += " FROM  "
        for tuple_name in dc_object.tuple_names:
            query += self.dataset.table_specific_name("Init") + \
                 " as " + tuple_name + ","
        query = query[:-1]
        query += " WHERE "
        if len(dc_object.tuple_names) == 2:
            query += dc_object.tuple_names[0] + "." + self.index + \
                 " != " + dc_object.tuple_names[1] + "." + self.index + " AND "
        query += dc_object.cnf_form
        self.dataengine.query(query)

        t4 = time.time()
        if self.holo_obj.verbose:
            self.holo_obj.logger.\
                info("Time for executing query " + dc_name + ":" + str(t4-t3))

        # For each predicate add attributes
        tuple_attributes = {}
        for tuple_name in dc_object.tuple_names:
            tuple_attributes[tuple_name] = set()

        for predicate in dc_object.predicates:

            for component in predicate.components:
                if isinstance(component, str):
                    pass
                else:
                    tuple_attributes[component[0]].add(component[1])

        tuple_attributes_lists = {}
        tuple_attributes_dfs = {}
        for tuple_name in dc_object.tuple_names:
            tuple_attributes_lists[tuple_name] = [[i] for i in
                                                  tuple_attributes[tuple_name]]
            tuple_attributes_dfs[
                tuple_name] = self.spark_session.createDataFrame(
                tuple_attributes_lists[tuple_name], ['attr_name'])

            name = self.dataset.table_specific_name(tuple_name + "_attributes")
            attribute_dataframe = tuple_attributes_dfs[tuple_name]

            self.dataengine.dataframe_to_table(name, attribute_dataframe)

            distinct = \
                "(SELECT  " + tuple_name + "_ind " \
                                                   " FROM " + \
                temp_table + ") AS row_table GROUP BY ind, attr"

            query = "INSERT INTO " + \
                    self.dataset.table_specific_name(table_id) + \
                    " SELECT row_table. " + tuple_name + "_ind as ind," \
                    " a.attr_name as attr, count(*) as count FROM " + \
                    name + \
                    " AS a," + \
                    distinct
            self.dataengine.query(query)
            df = self.dataengine.get_table_to_dataframe(table_id, self.dataset)
            self.holo_obj.logger.info('Denial Constraint Query Left ' +
                                      dc_name + ":" + query)
            drop_temp_table = "DROP TABLE " + name
            self.dataengine.query(drop_temp_table)
        drop_temp_table = "DROP TABLE " + temp_table
        self.dataengine.query(drop_temp_table)
        return df

    def _get_sym_noisy_cells_for_dc(self, dc_name):
        """
        Returns a dataframe that consists of index of noisy cells index,
        attribute

        :param dc_name: denial constraint
        :return: spark_dataframe
        """

        self.holo_obj.logger.info('Denial Constraint Queries For ' + dc_name)
        temp_table = "tmp" + self.dataset.dataset_id
        query = "CREATE TABLE " + \
                temp_table + " AS SELECT " \
                             "t1." + self.index + \
                " as t1_ind, " \
                "t2." + self.index + \
                " as t2_ind " \
                " FROM  " + \
                self.dataset.table_specific_name("Init") + \
                " as t1, " + \
                self.dataset.table_specific_name("Init") + \
                " as  t2 " + "WHERE t1." + self.index + \
                " != t2." + self.index + \
                "  AND " + dc_name
        self.dataengine.query(query)

        t1_attributes = set()

        dc_predicates = self.dictionary_dc[dc_name]
        for predicate_index in range(0, len(dc_predicates)):
            predicate_type = dc_predicates[predicate_index][4]
            # predicate_type 0 : we do not have a literal in this predicate
            # predicate_type 1 : literal on the left side of the predicate
            # predicate_type 2 : literal on the right side of the predicate
            if predicate_type == 0:
                relax_indices = range(2, 4)
            elif predicate_type == 1:
                relax_indices = range(3, 4)
            elif predicate_type == 2:
                relax_indices = range(2, 3)
            else:
                raise ValueError(
                    'predicate type can only be 0: '
                    'if the predicate does not have a literal'
                    '1: if the predicate has a literal in the left side,'
                    '2: if the predicate has a literal in right side'
                )
            for relax_index in relax_indices:
                name_attribute = \
                    dc_predicates[predicate_index][relax_index].split(".")
                if name_attribute[0] == "t1":
                    t1_attributes.add(name_attribute[1])

        left_attributes = [[i] for i in t1_attributes]

        t1_attributes_dataframe = self.spark_session.createDataFrame(
            left_attributes, ['attr_name'])

        t1_name = self.dataset.table_specific_name("T1_attributes")
        self.dataengine.dataframe_to_table(t1_name, t1_attributes_dataframe)

        # Left part of predicates
        distinct_left = \
            "(SELECT t1_ind  FROM " + temp_table + ") AS row_table"

        query_left = "INSERT INTO " + \
                     self.dataset.table_specific_name("C_dk_temp") + \
                     " SELECT row_table.t1_ind as ind," \
                     " a.attr_name as attr FROM " + \
                     t1_name + \
                     " AS a," + \
                     distinct_left
        df = self.dataengine.query(query_left, 1)
        self.holo_obj.logger.info('Denial Constraint Query Left ' +
                                  dc_name + ":" + query_left)

        drop_temp_table = "DROP TABLE " + temp_table
        self.dataengine.query(drop_temp_table)
        return df

    # Getters
    def get_noisy_cells(self):
        """
        Returns a dataframe that consists of index of noisy cells index,
         attribute

        :return: spark_dataframe
        """


        df = None
        table_ids = []
        for (idx, dc_name) in enumerate(self.dc_objects):
            table_id = "C_dk_temp" + str(idx)
            table_ids.append(table_id)
            table_name = self.dataset.table_specific_name(table_id)
            query_for_creation_table = "CREATE TABLE " + table_name + \
                                       "(ind INT, attr VARCHAR(255), count INT);"
            self.dataengine.query(query_for_creation_table)
            sub_df = self._get_noisy_cells_for_dc(dc_name,table_id)
            if df is None:
                df = sub_df.withColumnRenamed("count", "count{}".format(idx))

            else:
                df = df.join(sub_df.withColumnRenamed("count","count{}".format(idx)),
                                    ["ind", "attr"],"outer")
        self.dfs = df.na.fill(0).distinct()
        self.dataengine.dataframe_to_table(self.dataset.table_specific_name("DC"),self.dfs)
        self.dataengine.query("CREATE INDEX row_index ON {} (ind)".format(self.dataset.table_specific_name("DC")))
        return self.dfs


    def forward(self, example):
        if self.cache is None:
            query_str = "SELECT * FROM {}".format(self.dataset.table_specific_name("DC"))
            self.cache = sqlio.read_sql_query(query_str, self.conn)
        index, attr, val = example
        ret = torch.zeros(len(self.Denial_constraints))
        ans = self.cache[(self.cache["ind"] == int(index)) & (self.cache["attr"] == attr)]
        ans = ans.drop(["ind", "attr"], axis=1)
        if ans.empty:
            return ret
        else:
            df_unpack = ans.values
            # ret = numpy.array(df_unpack)
            ret = torch.from_numpy(df_unpack).squeeze()
        return ret




