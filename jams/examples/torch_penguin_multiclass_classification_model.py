import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
import json


# seed function for reproducibility
def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed(113)

"""
SECTION 1 : Load and setup data for training

the datasets separated in two files from originai datasets:
penguin-clean-train.csv = datasets for training purpose, 70% from the original data
penguin-clean-test.csv  = datasets for testing purpose, 30% from the original data
"""

# Section 1.1 Data Loading

# load
datatrain = pd.read_csv('datasets/penguins-clean-train.csv')

# Section 1.2 Preprocessing

# change string value to numeric
datatrain.loc[datatrain['species'] == 'Adelie', 'species'] = 0
datatrain.loc[datatrain['species'] == 'Gentoo', 'species'] = 1
datatrain.loc[datatrain['species'] == 'Chinstrap', 'species'] = 2
datatrain = datatrain.apply(pd.to_numeric)

# change dataframe to array
datatrain_array = datatrain.values

# split x and y (feature and target)
xtrain = datatrain_array[:, 1:]
ytrain = datatrain_array[:, 0]

"""
SECTION 2 : Build and Train Model

Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature from Palmer Penguin dataset
hidden layer : 20 neuron, activation using ReLU
output layer : 3 neuron, represents the number of species, Softmax Layer

optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.01
epoch = 50
"""

# hyperparameters
hl = 20
lr = 0.01
num_epoch = 50


# build model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

# choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

X = torch.Tensor(xtrain).float()
Y = torch.Tensor(ytrain).long()

# train
for epoch in range(num_epoch):
    # feedforward - backprop
    optimizer.zero_grad()
    out = net(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()
    acc = 100 * torch.sum(Y == torch.max(out.data, 1)[1]).double() / len(Y)
    print('Epoch [%d/%d] Loss: %.4f   Acc: %.4f'
          % (epoch + 1, num_epoch, loss.item(), acc.item()))

"""
SECTION 3 : Testing model
"""
# load
datatest = pd.read_csv('datasets/penguins-clean-test.csv')
print(datatest.head())
print(datatest.shape)

# change string value to numeric
datatest.loc[datatest['species'] == 'Adelie', 'species'] = 0
datatest.loc[datatest['species'] == 'Gentoo', 'species'] = 1
datatest.loc[datatest['species'] == 'Chinstrap', 'species'] = 2
datatest = datatest.apply(pd.to_numeric)

# change dataframe to array
datatest_array = datatest.values

# split x and y (feature and target)
xtest = datatest_array[:, 1:]
ytest = datatest_array[:, 0]


# get prediction
X = torch.Tensor(xtest).float()
Y = torch.Tensor(ytest).long()
out = net(X)
_, predicted = torch.max(out.data, 1)

# get accuration
print('Accuracy of the network %.4f %%' % (100 * torch.sum(Y == predicted).double() / len(Y)))

"""
SECTION 4 : Save the artefacts

load test data set and convert it to json for input.
save the json input and the model
"""
datatest.drop(['species'], axis=1, inplace=True)
sample_input = datatest.head(10).to_dict(orient='list')
with open("torch_input.json", "w") as outfile:
    json.dump(sample_input, outfile)
script_module = torch.jit.script(net)
script_module.save("torch_penguin.pt")
