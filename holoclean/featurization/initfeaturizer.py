import torch
from featurizer import Featurizer
from torch.nn import ParameterList


class InitFeaturizer(Featurizer):

    def __init__(self, N, L, session, update_flag=False):
        """
        Creates a pytorch module which will be a featurizer for HoloClean
        :param N : number of random variables
        :param L: number of classes
        :param session: HoloClean session
        :param update_flag: True if the values in tensor of the featurizer
        need be updated
        """
        super(InitFeaturizer, self).__init__(N, L, update_flag)
        self.session = session
        self.dataset = self.session.dataset
        self.dataengine = self.session.holo_env.dataengine
        self.table_name = self.dataset.table_specific_name('Init')
        self.id = "SignalInit"
        self.type = 1
        self.M = 1
        self.tensor = None

        if not self.update_flag:
            self.tensor_train = self.create_tensor(1, self.N, self.L)

        self.parameters = ParameterList()

    def create_tensor(self, clean=1, N=None, L=None):
        """
        This method creates the tensor for the feature
        """
        self.execute_query(clean)
        tensor = torch.zeros(N, self.M, L)

        query = "SELECT * FROM " + self.table_name
        feature_table = self.dataengine.query(query, 1).collect()
        for factor in feature_table:
            tensor[factor.vid - 1, factor.feature - 1,
                   factor.assigned_val - 1] = factor['count']
        return tensor

    def execute_query(self, clean=1):
        """
        Creates a string for the query that it is used to create the Initial
        Signal

        :param clean: shows if create the feature table for the clean or
        the don't know cells

        :return a list of length 1 with a string with the query
        for this feature
        """
        if clean:
            self.offset = self.session.feature_count
        count = self.offset + 1
        if clean:
            name = "Observed_Possible_values_clean"
        else:
            name = "Observed_Possible_values_dk"

        query_for_featurization = " SELECT  \
            init_flat.vid as vid, init_flat.domain_id AS assigned_val, \
            '" + str(1) + "' AS feature, \
            1 as count\
            FROM """ + \
            self.dataset.table_specific_name(name) + \
            " AS init_flat WHERE vid IS NOT NULL"

        # if clean add signal fo Feature_id_map
        if clean:
            self.session.feature_count += count

            index = self.offset + count
            list_domain_map = [[index, 'Init', 'Init', 'Init']]
            df_domain_map = self.session.holo_env.spark_session.\
                createDataFrame(list_domain_map,
                                self.dataset.attributes['Feature_id_map'])
            self.session.holo_env.dataengine.add_db_table(
                'Feature_id_map', df_domain_map, self.dataset, append=1)

        table_name = self.id + str(clean)
        self.table_name = self.dataset.table_specific_name(table_name)
        query_for_table = "CREATE TABLE " + self.table_name + \
                                  "(vid INT, assigned_val INT," \
                                  " feature INT ,count INT);"
        self.dataengine.query(query_for_table)
        self.dataengine.query("INSERT INTO " + self.table_name + "(" + query_for_featurization + ");")







