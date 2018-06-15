import sys
sys.path.append("..")
from holoclean.featurization.fast_text_gru import FastTextGRU
import pandas
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import string
import random
import numpy as np


seq_len_global = 20
class PandasDataset(Dataset):
    def __init__(self, df):
        self.df = df
        print(self.df.head())

    def __getitem__(self, index):
        return self.df.iloc[index].to_dict()

    def __len__(self):
        return self.df.shape[0]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.GRU = FastTextGRU(FastTextGRU.USE_PRETRAINED, seq_len_global, "wiki.en.bin")
        self.linear = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,words):
        X, h = self.GRU.forward(words)
        X_pad = self.manually_pad(X.squeeze(), 100)
        X = self.linear(X_pad)
        return self.sigmoid(X)

    def manually_pad(self, tensor, desired_len):
        diff = desired_len -tensor.size()[0]
        app = torch.zeros(diff)
        return torch.cat((tensor, app),0)

    def load_examples(self, examples):
        self.GRU.load_data(examples)

df = pandas.read_csv("words.txt")
df_clean = df.sample(frac=.25)
print('{} clean samples'.format(df_clean.shape[0]))
labels = pandas.Series([1])
label_frame = pandas.DataFrame({"label": labels})
df_clean = df_clean.assign(foo=1).merge(label_frame.assign(foo=1)).drop('foo', 1)
print(df_clean.head())
# now get some unclean samples
labels = pandas.Series([0])
label_frame = pandas.DataFrame({"label": labels})
df_dirty = df.sample(frac=.25)
df_dirty = df_dirty.assign(foo=1).merge(label_frame.assign(foo=1)).drop('foo', 1)



def add_typo(word):
    word = list(word)
    n = len(word)
    idx = random.choice(range(n))
    new_letter = random.choice(string.letters.lower())
    if word[idx] == new_letter:
        return add_typo(word)
    else:
        word[idx] = new_letter
        return ''.join(word)


print(add_typo("word"))
# now inject noise into the dirty rows
for index, row in df_dirty.iterrows():
    df_dirty.at[index, "word"] = add_typo(row["word"])
df_final = pandas.concat([df_dirty, df_clean])

train_data = df_final.sample(frac=.25)
test_data = df_final

data_loader = DataLoader(PandasDataset(train_data), batch_size=10,  shuffle=True)

def sort_by_tuples(text, labels):
    together = []
    for idx,word in enumerate(text):
        together.append((word,labels[idx]))
    sorted_list = sorted(together, key = lambda x: len(x[0]), reverse=True)
    text = []
    labels = []
    for (word, label) in sorted_list:
        text.append(word)
        labels.append(label)
    return text, labels


model = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.3)
for epoch in range(3):gs
    tot = 0.0
    cnt = len(data_loader)
    for (i_batch, data) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = np.array([])
        labels = np.array([])
        for idx in range(data_loader.batch_size):
            features = data['word'][idx]
            label = data['label'][idx]
            output = model.forward(features)
            loss = criterion(output, label.float())
            tot += loss.item()
            loss.backward()
        optimizer.step()
        print("batch {}: loss: {}".format(i_batch, loss.item()))
    print("epoch {}: avg loss {}".format(epoch, tot / cnt))
