import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



#make helper functions

def plot_digit(X, y, idx):
    img = X[idx].reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

classes = training_data.classes
test_classes = test_data.classes
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

c = input("What part do you want to test? (A/B/C/D): ")[0]

if c == "A":
    #Create test train datasets
    train_dataloader = DataLoader(training_data, batch_size=512)
    test_dataloader = DataLoader(test_data, batch_size=512)

    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.encode = nn.Sequential(
                nn.Linear(784, 100, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(100, 2, bias=True),
            )
            self.decode = nn.Sequential(
                nn.Linear(2, 100, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(100, 784, bias=True)
            )
        def forward(self, x):
            x = x.view(xbatch.size(0), -1)
            encoded = self.encode(x)
            logits = self.decode(encoded)
            return logits, encoded

    model = AutoEncoder()

    # binary cross entropy loss
    loss_fn = nn.MSELoss(reduction='sum')

    nEpochs = 50
    batch_size = 512

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    accuracy = np.zeros(nEpochs)
    for epoch in range(nEpochs):
        print(epoch)
        estimatedLabels = np.empty((0,))
        trueLabels = np.empty((0,))
        totloss = 0
        for xbatch, _ in train_dataloader:
            model.to(device)
            xbatch = xbatch.to(device)

            y_pred, encoded = model(xbatch)

            loss = loss_fn(y_pred, xbatch.view(xbatch.size(0), -1))

            # zero the gradient before running the backward pass
            model.zero_grad()

            loss.backward()
            totloss += loss
            optimizer.step()

            y_pred = y_pred.detach().cpu().numpy()
            xbatch = xbatch.detach().cpu().numpy()
            # calculate accuracy for whole training set
            yhat = y_pred > 0.5
            estimatedLabels = np.append(estimatedLabels, yhat)
            trueLabels = np.append(trueLabels, xbatch)
        accuracy[epoch] = np.mean(estimatedLabels == trueLabels)
        print("Tot loss: ", (totloss/len(train_dataloader)))
    print("Mean accuracy: ", np.mean(accuracy))
    print("Accuracy: ", accuracy)

    # test performance
    for xbatch, true_labels_batch in test_dataloader:
        model.to(device)
        xbatch = xbatch.to(device)
        y_pred, encoded = model(xbatch)

        # calculate accuracy for whole test set

        y_pred = y_pred.detach().cpu().numpy()
        encoded = encoded.detach().cpu().numpy()
        xbatch = xbatch.detach().cpu().numpy()
        yhat = y_pred > 0.5
        true_labels_batch = true_labels_batch.detach().cpu().numpy()
        plt.scatter(encoded[:, 0], encoded[:, 1], c=true_labels_batch, cmap='tab10', alpha=0.6)
    plt.show()

elif c == "B":
    train_dataloader = DataLoader(training_data, batch_size=512)
    test_dataloader = DataLoader(test_data, batch_size=512)

    class VariationalAutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(784, 100)
            self.fc_mu = nn.Linear(100, 2)
            self.fc_logvar = nn.Linear(100, 2)
            self.decoder = nn.Sequential(
                nn.Linear(2, 100),
                nn.ReLU(),
                nn.Linear(100, 784)
            )

        def encode(self, x):
            h = nn.functional.relu(self.fc1(x))
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decode(z)
            return reconstructed, mu, logvar, z


    def vae_loss(reconstructed, original, mu, logvar):
        reconstruction_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence


    model = VariationalAutoEncoder()

    # binary cross entropy loss
    loss_fn = nn.KLDivLoss()

    nEpochs = 50
    batch_size = 512

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    for epoch in range(nEpochs):
        model.train()
        total_loss = 0
        for xbatch, _ in train_dataloader:
            xbatch = xbatch.to(device)
            optimizer.zero_grad()
            reconstructed, mu, logvar, _ = model(xbatch)
            loss = vae_loss(reconstructed, xbatch.view(xbatch.size(0), -1), mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_dataloader)}")

    model.eval()
    with torch.no_grad():
        for xbatch, labels in test_dataloader:
            xbatch = xbatch.to(device)
            _, _, _, z = model(xbatch)
            z = z.cpu().numpy()
            labels = labels.numpy()
            plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar()
    plt.title("Latent Space of VAE")
    plt.show()

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
            #self.inputLayer = nn.Linear(784, 64, bias=True)
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

