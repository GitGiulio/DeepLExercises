#load packages
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
import torch
from torch import nn
from mlxtend.data import mnist_data


#make helper functions

def plot_digit(X, y, idx):
    img = X[idx].reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()

#load data
X, y = mnist_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

c = input("What part do you want to test? (A/B/C/D): ")[0]

if c == "A":
    #we just focus on 1's and 0's:
    keep = (y == 0) | (y == 1)
    X = X[keep, :]
    y = y[keep]
    #Create test train datasets
    Xtrain,Xtest,ytrain,ytest = sklearn.model_selection.train_test_split(X,y,test_size=0.3,shuffle=True)
    #transform into pytorch datasets
    trainDataset = torch.utils.data.TensorDataset(torch.Tensor(Xtrain),torch.Tensor(ytrain))
    testDataset = torch.utils.data.TensorDataset(torch.Tensor(Xtest),torch.Tensor(ytest))

    class LogisticRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputLayer = nn.Linear(784, 1, bias=True)
        def forward(self, x):
            x = self.inputLayer(x)
            x = torch.sigmoid(x)
            return torch.squeeze(x)

    model = LogisticRegression()

    # binary cross entropy loss
    loss_fn = nn.BCELoss()

    nEpochs = 10
    learning_rate = 0.001
    batch_size = 64

    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    accuracy = np.zeros(nEpochs)
    for epoch in range(nEpochs):
        estimatedLabels = np.empty((0,))
        trueLabels = np.empty((0,))
        for xbatch, ybatch in trainDataLoader:
            model.to(device)
            xbatch.to(device)
            ybatch.to(device)

            y_pred = model(xbatch)

            loss = loss_fn(y_pred, ybatch)

            # zero the gradient before running the backward pass
            model.zero_grad()

            loss.backward()

            optimizer.step()

            y_pred.detach().to('cpu').numpy()
            # calculate accuracy for whole training set
            yhat = y_pred > 0.5
            estimatedLabels = np.append(estimatedLabels, yhat)
            trueLabels = np.append(trueLabels, ybatch)
        accuracy[epoch] = np.mean(estimatedLabels == trueLabels)
    print("Mean accuracy: ", np.mean(accuracy))
    print("Accuracy: ", accuracy)

    # test performance
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False)
    estimatedLabels = np.empty((0,))
    trueLabels = np.empty((0,))
    for xbatch, ybatch in testDataLoader:
        model.to(device)
        xbatch.to(device)
        ybatch.to(device)
        y_pred = model(xbatch)
        # calculate accuracy for whole test set
        yhat = y_pred > 0.5
        estimatedLabels = np.append(estimatedLabels, yhat)
        trueLabels = np.append(trueLabels, ybatch)
    print("Mean test accuracy: ", np.mean(estimatedLabels == trueLabels))

elif c == "B":
    # Create test train datasets
    Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X, y, test_size=0.3, shuffle=True)
    # transform into pytorch datasets
    trainDataset = torch.utils.data.TensorDataset(torch.Tensor(Xtrain), torch.LongTensor(ytrain))
    testDataset = torch.utils.data.TensorDataset(torch.Tensor(Xtest), torch.LongTensor(ytest))


    class MyModelFor10Digits(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputLayer = nn.Linear(784, 10, bias=True)

        def forward(self, x):
            x = self.inputLayer(x)
            x = torch.softmax(x,dim=1)
            return torch.squeeze(x)

    model = MyModelFor10Digits()

    # Here I use cross entropy because I have more than 2 classes (also for the project I will do so)
    loss_fn = nn.CrossEntropyLoss()

    nEpochs = 10
    learning_rate = 0.001
    batch_size = 64

    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    accuracy = np.zeros(nEpochs)
    for epoch in range(nEpochs):
        estimatedLabels = np.empty((0,))
        trueLabels = np.empty((0,))
        for xbatch, ybatch in trainDataLoader:
            y_pred = model(xbatch)

            loss = loss_fn(y_pred, ybatch)

            # zero the gradient before running the backward pass
            model.zero_grad()

            loss.backward()

            optimizer.step()

            # calculate accuracy for whole training set
            yhat = torch.argmax(y_pred, dim=1)
            estimatedLabels = np.append(estimatedLabels, yhat)
            trueLabels = np.append(trueLabels, ybatch)
        accuracy[epoch] = np.mean(estimatedLabels == trueLabels)
    print("Mean accuracy: ", np.mean(accuracy))
    print("Accuracy: ", accuracy)

    # test performance
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False)
    estimatedLabels = np.empty((0,))
    trueLabels = np.empty((0,))
    for xbatch, ybatch in testDataLoader:
        y_pred = model(xbatch)
        # calculate accuracy for whole test set
        yhat = torch.argmax(y_pred, dim=1)
        estimatedLabels = np.append(estimatedLabels, yhat)
        trueLabels = np.append(trueLabels, ybatch)
    print("Mean test accuracy: ", np.mean(estimatedLabels == trueLabels))
elif c == "C":
    # Create test train datasets
    Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X, y, test_size=0.3, shuffle=True)
    # transform into pytorch datasets
    trainDataset = torch.utils.data.TensorDataset(torch.Tensor(Xtrain), torch.LongTensor(ytrain))
    testDataset = torch.utils.data.TensorDataset(torch.Tensor(Xtest), torch.LongTensor(ytest))


    class MyModelFor10Digits(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputLayer = nn.Linear(784, 32, bias=True)
            self.hiddenLayer1 = nn.Linear(32, 32, bias=True)
            self.hiddenLayer2 = nn.Linear(32, 16, bias=True)
            self.hiddenLayer3 = nn.Linear(16, 16, bias=True)
            self.outputLayer = nn.Linear(16, 10, bias=True)

        def forward(self, x):
            x = self.inputLayer(x)
            x = torch.relu(x)
            x = self.hiddenLayer1(x)
            x = torch.relu(x)
            x = self.hiddenLayer2(x)
            x = torch.relu(x)
            x = self.hiddenLayer3(x)
            x = torch.relu(x)
            x = self.outputLayer(x)
            #x = torch.softmax(x, dim=1) already implemented in Cross entropy loss (so better not)
            return torch.squeeze(x)


    model = MyModelFor10Digits()

    # Here I use cross entropy because I have more than 2 classes (also for the project I will do so)
    loss_fn = nn.CrossEntropyLoss()

    nEpochs = 25
    learning_rate = 0.001
    batch_size = 64

    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.0005)

    accuracy = np.zeros(nEpochs)
    for epoch in range(nEpochs):
        estimatedLabels = np.empty((0,))
        trueLabels = np.empty((0,))
        for xbatch, ybatch in trainDataLoader:
            y_pred = model(xbatch)

            loss = loss_fn(y_pred, ybatch)

            # zero the gradient before running the backward pass
            model.zero_grad()

            loss.backward()

            optimizer.step()

            # calculate accuracy for whole training set
            yhat = torch.argmax(y_pred, dim=1)
            estimatedLabels = np.append(estimatedLabels, yhat)
            trueLabels = np.append(trueLabels, ybatch)
        accuracy[epoch] = np.mean(estimatedLabels == trueLabels)
    print("Mean accuracy: ", np.mean(accuracy))
    print("Accuracy: ", accuracy)

    # test performance
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False)
    estimatedLabels = np.empty((0,))
    trueLabels = np.empty((0,))
    for xbatch, ybatch in testDataLoader:
        y_pred = model(xbatch)
        # calculate accuracy for whole test set
        yhat = torch.argmax(y_pred, dim=1)
        estimatedLabels = np.append(estimatedLabels, yhat)
        trueLabels = np.append(trueLabels, ybatch)
    print("Mean test accuracy: ", np.mean(estimatedLabels == trueLabels))
elif c == "D":
    # Create test train datasets
    Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X, y, test_size=0.3, shuffle=True)
    # transform into pytorch datasets
    trainDataset = torch.utils.data.TensorDataset(torch.Tensor(Xtrain), torch.LongTensor(ytrain))
    testDataset = torch.utils.data.TensorDataset(torch.Tensor(Xtest), torch.LongTensor(ytest))


    class CNN10Digits(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputLayer = nn.Linear(784, 32, bias=True)
            self.conv1 = nn.Conv2d(1, 16, 3,padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3,padding=1)
            self.pool = nn.MaxPool2d(2,2)
            self.liner1 = nn.Linear(32 * 7 * 7, 128, bias=True)
            self.outputLayer = nn.Linear(128, 10, bias=True)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 32 * 7 * 7) # flatten
            x = torch.relu(self.liner1(x))
            x = self.outputLayer(x)
            # x = torch.softmax(x, dim=1)
            return torch.squeeze(x)


    model = CNN10Digits()

    # Here I use cross entropy because I have more than 2 classes (also for the project I will do so)
    loss_fn = nn.CrossEntropyLoss()

    nEpochs = 10
    learning_rate = 0.001
    batch_size = 64

    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accuracy = np.zeros(nEpochs)
    for epoch in range(nEpochs):
        estimatedLabels = np.empty((0,))
        trueLabels = np.empty((0,))
        for xbatch, ybatch in trainDataLoader:
            xbatch = xbatch.view(-1, 1, 28, 28)
            y_pred = model(xbatch)

            loss = loss_fn(y_pred, ybatch)

            # zero the gradient before running the backward pass
            model.zero_grad()

            loss.backward()

            optimizer.step()

            # calculate accuracy for whole training set
            yhat = torch.argmax(y_pred, dim=1)
            estimatedLabels = np.append(estimatedLabels, yhat)
            trueLabels = np.append(trueLabels, ybatch)
        accuracy[epoch] = np.mean(estimatedLabels == trueLabels)
    print("Mean accuracy: ", np.mean(accuracy))
    print("Accuracy: ", accuracy)

    # test performance
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False)
    estimatedLabels = np.empty((0,))
    trueLabels = np.empty((0,))
    for xbatch, ybatch in testDataLoader:
        xbatch = xbatch.view(-1, 1, 28, 28)
        y_pred = model(xbatch)
        # calculate accuracy for whole test set
        yhat = torch.argmax(y_pred, dim=1)
        estimatedLabels = np.append(estimatedLabels, yhat)
        trueLabels = np.append(trueLabels, ybatch)
    print("Mean test accuracy: ", np.mean(estimatedLabels == trueLabels))
else:
    raise ValueError("Invalid input")

