import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def DataWork(time, data):
    m, n = data.shape
    X = []
    Y = []
    for i in range(m - time):
        x = data[i: i + time, 1:]
        y = data[int(i + time), 0]
        X.append(x)
        Y.append(y)
    return torch.tensor(np.array(X).astype(np.float32)), torch.tensor(np.array(Y).astype(np.float32))


class myDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y.reshape(len(y), 1)

    def __getitem__(self, index):
        return self.x[index, :, :], self.y[index, :]

    def __len__(self):
        return self.x.shape[0]


class Net(nn.Module):
    def __init__(self, input_, hidden_, out_, batch_):
        super(Net, self).__init__()
        self.input_size = input_
        self.hidden_size = hidden_
        self.output_size = out_
        self.batch_size = batch_
        self.num_layers = 1
        self.LSTM = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.Linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.LSTM(x)
        out = self.Linear(out)
        out = out[:, -1, -1]
        return out


def Train(epochs, model, criterion, optimizer):
    loss_epoch = []
    for epoch in range(epochs):
        model.train()
        loss_ = 0
        for i, data in enumerate(DataTrain, 0):
            x, y = data
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss_ += loss.item()
            loss.backward()
            optimizer.step()
        loss_epoch.append(loss_)
        print(loss_)
    plt.figure(1)
    plt.plot(loss_epoch)
    plt.show()


def Test(model, data_test):
    Y = []
    Y_pre = []
    model.eval()
    with torch.no_grad():
        for j, data_ in enumerate(data_test, 0):
            x_, y_ = data_
            output = model(x_)
            Y.append(y_)
            Y_pre.append(output)
    Y_pre = scaler.fit_transform(np.array(Y_pre).reshape(-1, 1))
    return np.array(Y).reshape(-1, 1), np.array(Y_pre).reshape(-1, 1)


def DrawFig(Y, Pre):
    plt.figure(2)
    plt.plot(np.array(Y), color='red', label='y')
    plt.plot(np.array(Pre), color='blue', label='y_pre')
    plt.title('Prediction')
    plt.xlabel('Time')
    plt.ylabel('')
    plt.legend()
    plt.show()


batch_size = 10
seq_len = 2
trainCSV = pd.read_csv('datasets-writeable/archive/DailyDelhiClimateTrain.csv', index_col=0).values
testCSV = pd.read_csv('datasets-writeable/archive/DailyDelhiClimateTest.csv', index_col=0).values

scaler = MinMaxScaler(feature_range=(0, 1))
trainCSV = scaler.fit_transform(trainCSV)
testCSV = scaler.fit_transform(testCSV)
X_train, Y_train = DataWork(time=seq_len, data=trainCSV)
train = myDataset(X_train, Y_train)
DataTrain = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
X_test, Y_test = DataWork(time=seq_len, data=testCSV)
test = myDataset(X_test, Y_test)
DataTest = DataLoader(dataset=test, batch_size=1, shuffle=False)

model = Net(input_=3, hidden_=5, out_=1, batch_=batch_size)
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Train(epochs=100, model=model, criterion=criterion, optimizer=optimizer)
Y, Y_pre = Test(model=model, data_test=DataTest)
DrawFig(Y, Y_pre)
train_ = DataLoader(dataset=train, batch_size=1, shuffle=False)
Y, Y_pre = Test(model=model, data_test=train_)
DrawFig(Y, Y_pre)
print('end')
