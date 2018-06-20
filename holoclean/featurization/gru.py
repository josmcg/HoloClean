from holoclean.learning.fast_text_gru import FastTextGRU
from holoclean.featurization.featurizer import Featurizer


class GRUFeaturizer(Featurizer):
    def __init__(self, N, session, per_col=True, col_corpa=None ):
        """
        Initialize a GRU featurizer
        :param N: the number of random variables
        :param session: a holoclean session object
        :param per_col: True if a gru is to be trained per column
            if false, one gru will be trained for each column
        :param col_corpa: the Corpus to use for each embedding layer
            if given FastTextGRU.USE_PRETRAINED, it will use a pretrained model of
            english embeddings, otherwise it will use the column data.
        """
        super(GRUFeaturizer, self).__init__(N, 1, True)
        self.session = session
        self.per_col = per_col
        self.col_corpa = col_corpa
        if self.per_col:
            self.gru = []
        else:
            self.gru = self.create_model()

    def create_tensor(self, clean=1, N=None, L=None):
        """
        runs each cell through it's associated GRU module and then concatenates them
        :param clean: whether we are training on clean data or not
        :param N: the number of random variables
        :param L: the number of classes
        :return: the created tensor from the GRU outputs
        """
        pass

    def create_model(self, column, corpus):
        """

        :param column: The column the model is for
        :param corpus: the corpus to use
        :return: a FastTextGRU module
        """
        return FastTextGRU
