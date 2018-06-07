import torch.nn as nn
import torch
from holoclean.global_variables import GlobalVariables
from featurizermodule import Featurizer
from torch.nn import Parameter, ParameterList

class DCFeaturizer(Featurizer):

    def __init__(self, N, L, update_flag=False, session=None, denial_constraints=None):
        """
        Creates a pytorch module which will be a featurizer for HoloClean
        :param n : number of random variables
        :param l: number of classes
        :param f: number of features in the group
        :param update_flag: True if the values in tensor of the featurizer
        need be updated

        """
        super(DCFeaturizer, self).__init__(N, L, update_flag)
        self.M = None
        self.tensor = None
        self.id = "SignalDC"
        self.denial_constraints = denial_constraints
        self.session = session
        self.parser = session.parser
        self.dataset = self.session.dataset
        self.dataengine = self.session.holo_env.dataengine
        self.spark_session = self.session.holo_env.spark_session
        self.table_name = self.dataset.table_specific_name('Init')
        self.dc_objects = session.dc_objects
        if not self.update_flag:
            self.create_tensor()
        self.parameters = ParameterList()

    def create_tensor(self,clean=1, N=None, L=None):
        """
        This method creates the tensor for the feature
        """
        self.execute_query(clean)
        self.M = self.count
        if clean:
            tensor = torch.zeros(self.N, self.M, self.L)
        else:
            tensor = torch.zeros(N, self.M, L)

        query = "SELECT * FROM " + self.table_name
        feature_table = self.dataengine.query(query, 1).collect()
        for factor in feature_table:
            tensor[factor.vid - 1, factor.feature - 1,
                   factor.assigned_val - 1] = factor['count']
        self.tensor = tensor

        return self.tensor

    def _create_all_relaxed_dc(self):
        """
        This method creates a list of all the possible relaxed DC's

        :return: a list of all the possible relaxed DC's
        """
        all_relax_dc = []
        self.attributes_list = []
        for dc_name in self.dc_objects:
            relax_dcs = self._create_relaxed_dc(self.dc_objects[dc_name])
            for relax_dc in relax_dcs:
                all_relax_dc.append(relax_dc)
        return all_relax_dc

    def _create_relaxed_dc(self, dc_object):
        """
        This method creates a list of all the relaxed DC's for a specific DC

        :param dc_object: The dc object that we want to relax

        :return: A list of all relaxed DC's for dc_name
        """
        relax_dcs = []
        index_name = GlobalVariables.index_name
        dc_predicates = dc_object.predicates
        for predicate in dc_predicates:
            component1 = predicate.components[0]
            component2 = predicate.components[1]
            full_form_components = \
                predicate.cnf_form.split(predicate.operation)
            if not isinstance(component1, str):
                self.attributes_list.append(component1[1])
                relax_dc = "postab.tid = " + component1[0] + \
                           "." + index_name + " AND " + \
                           "postab.attr_name = '" + component1[1] + \
                           "' AND " + "postab.attr_val"   \
                           + predicate.operation + \
                           full_form_components[1]

                if len(dc_object.tuple_names) > 1:
                    if isinstance(component1, list) and isinstance(
                            component2, list):

                        if component1[1] != component2[1]:
                            relax_dc = relax_dc + " AND  " + \
                                       dc_object.tuple_names[0] + "." + \
                                       index_name + \
                                       " <> " + dc_object.tuple_names[1] + "."\
                                       + index_name
                        else:
                            relax_dc = relax_dc + " AND  " + \
                                       dc_object.tuple_names[0] + "." + \
                                       index_name \
                                       + " < " + dc_object.tuple_names[
                                           1] + "." + \
                                       index_name
                    else:
                        relax_dc = relax_dc + " AND  " + dc_object.tuple_names[
                            0] + "." + \
                                   index_name \
                                   + " < " + dc_object.tuple_names[1] + "." + \
                                   index_name

                for other_predicate in dc_predicates:
                    if predicate != other_predicate:
                        relax_dc = relax_dc + " AND  " + \
                                   other_predicate.cnf_form
                relax_dcs.append([relax_dc, dc_object.tuple_names])

            if not isinstance(component2, str):
                self.attributes_list.append(component2[1])
                relax_dc = "postab.tid = " + component2[0] +\
                           "." + index_name + " AND " + \
                           "postab.attr_name ='" + component2[1] +\
                           "' AND " + full_form_components[0] + \
                           predicate.operation + \
                           "postab.attr_val"
                if len(dc_object.tuple_names) > 1:
                    if isinstance(component1, list) and isinstance(
                            component2, list):
                        if component1[1] != component2[1]:
                            relax_dc = relax_dc + " AND  " + \
                                       dc_object.tuple_names[0] + "." + \
                                       index_name + \
                                       " <> " + dc_object.tuple_names[1] \
                                       + "." + index_name
                        else:
                            relax_dc = relax_dc + " AND  " + \
                                       dc_object.tuple_names[0] + "." + \
                                       index_name \
                                       + " < " + dc_object.tuple_names[
                                           1] + "." + \
                                       index_name
                    else:
                        relax_dc = relax_dc + " AND  " + dc_object.tuple_names[
                            0] + "." + \
                                   index_name \
                                   + " < " + dc_object.tuple_names[1] + "." + \
                                   index_name

                for other_predicate in dc_predicates:
                    if predicate != other_predicate:
                        relax_dc = relax_dc + " AND  " + \
                                   other_predicate.cnf_form
                relax_dcs.append([relax_dc, dc_object.tuple_names])

        return relax_dcs

    def execute_query(self,clean):
        """
        Creates a list of strings for the queries that are used to create the
        DC Signals

        :param clean: shows if we create the feature table for the clean or the
        dk cells
        :param dcquery_prod: a thread that we will produce the final queries

        :return a list of strings for the queries for this feature
        """
        if clean:
            name = "Possible_values_clean"
        else:
            name = "Possible_values_dk"

        all_relax_dcs = self._create_all_relaxed_dc()
        dc_queries = []
        count = 0
        if clean:
            self.offset = self.session.feature_count

        feature_map = []
        for index_dc in range(0, len(all_relax_dcs)):
            relax_dc = all_relax_dcs[index_dc][0]
            table_name = all_relax_dcs[index_dc][1]
            count += 1
            query_for_featurization = "SELECT" \
                                      " postab.vid as vid, " \
                                      "postab.domain_id AS assigned_val, " + \
                                      str(count) \
                                      + " AS feature, " \
                                      "  count(*) as count " \
                                      "  FROM "
            for tuple_name in table_name:
                query_for_featurization += \
                    self.dataset.table_specific_name("Init") + \
                    " as " + tuple_name + ","
            query_for_featurization += \
                self.dataset.table_specific_name(name) + " as postab"

            query_for_featurization += " WHERE " + relax_dc + \
                                       " GROUP BY postab.vid, postab.domain_id"
            dc_queries.append(query_for_featurization)

            if clean:
                feature_map.append([count + self.offset,
                                    self.attributes_list[index_dc],
                                    relax_dc, "DC"])

                df_feature_map_dc = self.spark_session.createDataFrame(
                    feature_map, self.dataset.attributes['Feature_id_map'])
                self.dataengine.add_db_table('Feature_id_map',
                                             df_feature_map_dc, self.dataset, 1)
                self.session.feature_count += count


        self.count = len(dc_queries)
        table_name = self.id + str(clean)
        self.table_name = self.dataset.table_specific_name(table_name)
        query_for_table = "CREATE TABLE " + self.table_name + \
                                  "(vid INT, assigned_val INT," \
                                  " feature INT ,count INT);"
        self.dataengine.query(query_for_table)

        #execute dc_queries
        for query in dc_queries:
            self.dataengine.query(
                "INSERT INTO " + self.table_name + "(" + query + ");")


        return