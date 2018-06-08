import torch.nn as nn
import torch
from holoclean.global_variables import GlobalVariables
from featurizer import Featurizer
from torch.nn import ParameterList

class CooccurFeaturizer(Featurizer):

    def __init__(self, N, L, session):
        """
        Creates a pytorch module which will be a featurizer for HoloClean
        :param N : number of random variables
        :param L: number of classes
        :param session: HoloClean session
        :param update_flag: True if the values in tensor of the featurizer
        need be updated
        """
        super(CooccurFeaturizer, self).__init__(N, L, False)
        self.session = session
        self.offset = self.session.feature_count
        self.index_name = GlobalVariables.index_name
        self.all_attr = list(self.session.init_dataset.schema.names)
        self.all_attr.remove(self.index_name)
        self.count = 0
        self.pruning_object = self.session.pruning
        self.domain_pair_stats = self.pruning_object.domain_pair_stats
        self.dirty_cells_attributes = \
            self.pruning_object.dirty_cells_attributes
        self.domain_stats = self.pruning_object.domain_stats
        self.threshold = self.pruning_object.threshold1
        self.direct_insert = True
        self.dataengine = self.session.holo_env.dataengine
        self.update_flag = False
        self.get_feature_id_map()
        self.id = "SignalCooccur"
        self.type = 1
        self.M = self.count
        self.attribute_feature_id = None
        self.tensor = None
        if not self.update_flag:
            self.create_tensor()
        self.parameters = ParameterList()

    def create_tensor(self, clean=1, N=None, L=None):
        """
        This method creates the tensor for the feature
        """

        domain_pair_stats = self.pruning_object.domain_pair_stats
        domain_stats = self.pruning_object.domain_stats
        cell_domain = self.pruning_object.cell_domain
        cell_values = self.pruning_object.cellvalues

        if clean:
            vid_list = self.pruning_object.v_id_clean_list
            tensor = torch.zeros(self.N, self.M, self.L)

        else:
            vid_list = self.pruning_object.v_id_dk_list
            tensor = torch.zeros(N, self.M, L)

        for vid in range(len(vid_list)):
            for cell_index in cell_values[vid_list[vid][0] - 1]:

                co_attribute = \
                    cell_values[vid_list[vid][0] - 1][cell_index].columnname
                attribute = vid_list[vid][1]
                feature = self.attribute_feature_id.get(co_attribute, -1)

                if co_attribute != attribute and feature != -1:
                    domain_id = 0
                    co_value = \
                        cell_values[vid_list[vid][0] - 1][cell_index].value

                    for value in cell_domain[vid_list[vid][2]]:
                        v_count = domain_stats[co_attribute][co_value]
                        count = domain_pair_stats[co_attribute][attribute].get(
                            (co_value, value), 0)
                        probability = count / v_count
                        tensor[vid, feature - 1, domain_id] = probability
                        domain_id = domain_id + 1
        self.tensor = tensor

        return self.tensor

    def get_feature_id_map(self):
        """
        Adding co-occur feature
        :param clean: shows if create the feature table for the clean or the dk
         cells
        :return list
        """

        self.offset = self.session.feature_count
        self.attribute_feature_id = {}
        feature_id_list = []
        for attribute in self.dirty_cells_attributes:
            self.count += 1
            self.attribute_feature_id[attribute] = self.count
            feature_id_list.append(
                [self.count + self.offset, attribute, 'Cooccur',
                 'Cooccur'])
            feature_df = self.session.holo_env.spark_session.createDataFrame(
                feature_id_list,
                self.session.dataset.attributes['Feature_id_map']
            )
            self.dataengine.add_db_table(
                'Feature_id_map',
                feature_df,
                self.session.dataset,
                append=1
            )
            self.session.feature_count += self.count
        return
