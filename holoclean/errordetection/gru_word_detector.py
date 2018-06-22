import torch.nn as nn
from fastText import load_model, train_unsupervised
import torch
from torch.nn.modules.sparse import EmbeddingBag
import numpy as np
from torch.autograd import Variable
import os.path


class FastTextEmbeddingBag(EmbeddingBag):
    def __init__(self, model_path):
        self.model = load_model(model_path)
        input_matrix = self.model.get_input_matrix()
        input_matrix_shape = input_matrix.shape
        super(FastTextEmbeddingBag, self).__init__(input_matrix_shape[0], input_matrix_shape[1])
        self.weight.data.copy_(torch.FloatTensor(input_matrix))

    def forward(self, words):
        word_subinds = np.empty([0], dtype=np.int64)
        word_offsets = [0]

        for word in words:
            _, subinds = self.model.get_subwords(word)
            word_subinds = np.concatenate((word_subinds, subinds))
            word_offsets.append(word_offsets[-1] + len(subinds))
        word_offsets = word_offsets[:-1]
        ind = Variable(torch.LongTensor(word_subinds))
        offsets = Variable(torch.LongTensor(word_offsets))
        return super(FastTextEmbeddingBag, self).forward(ind, offsets).reshape(-1, 1, 100)


class FastTextGRU(nn.Module):
    USE_PRETRAINED = 0
    def __init__(self, corpus, seq_len, path=None):
        """
        Initializes a FastText embedding that is fed into a GRU
        :param N: number of examples
        :param session: a HoloClean Session object
        :param corpus: the corpus the FastText Embbeding will be built from
        :param seq_len: the length of sequence that will be fed into the GRU
            necessary so padding can be done to make all sequences fit
        """
        super(FastTextGRU, self).__init__()
        self.corpus = corpus
        self.path = path
        self.count = seq_len
        self.embedding = self.get_embedding()
        # TODO how do we control the # of features per embedding?
        self.gru_feats = 100
        self.gru = nn.GRU(self.gru_feats, self.count, 1)
        self.embed_vecs = None

    def get_embedding(self):
        if self.corpus == self.USE_PRETRAINED:
            return FastTextEmbeddingBag(self.path)
        # the corpus must then be a list of words
        assert isinstance(self.corpus, type([]))
        file_path = "tmp/embbeding_corpora.csv"
        with open(file_path, "w") as handle:
            for item in self.corpus:
                handle.write("{}\n".format(item))
        # train the unsupervised representation
        embedding = train_unsupervised(os.path.abspath(file_path)
                                       , model="skipgram")
        embedding_path = "tmp/embedding.bin"
        embedding.save_model(embedding_path)
        return FastTextEmbeddingBag(embedding_path)


    def forward(self, example):
        """
        Foward method of the FastText GRU
        !!Important, this only supports batch sizes of size 1 at the moment
        :param example:
        :return:
        """
        index, attr, val = example
        val = val.decode("utf-8")
        val = unicode(val)
        embeddings = self.embedding(val.split())
        embeddings = embeddings.reshape(-1,  1, self.gru_feats)
        x, h = self.gru(embeddings)
        return x.squeeze()[len(val.split()) -1, :]






