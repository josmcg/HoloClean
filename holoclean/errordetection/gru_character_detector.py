from torch import nn
import torch
from torch.autograd import Variable

class GRUCharacterDetector(nn.Module):
    def __init__(self, cols):
        super(GRUCharacterDetector, self).__init__()
        self.count = 10
        self.n_letters = 256
        self.grus = {}
        for col in cols:
            self.grus[col] = nn.GRU(256, self.count, 1)

    def forward(self, example):
        index, attr, val = example
        gru = self.grus[attr]
        val = val.decode("utf-8")
        val = unicode(val)
        x = self.lineToTensor(val)
        fx, h = gru(x)
        return fx.squeeze()[len(val)-1, :]

    def letterToIndex(self, letter):
        return ord(letter)


    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

