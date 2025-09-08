#load packages
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt

#make helper functions

def plot_digit(X, y, idx):
    img = X[idx].reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()


#load data

from mlxtend.data import mnist_data

X, y = mnist_data()

#we just focus on 1's and 0's:
keep = (y == 0) | (y == 1)
X = X[keep, :]
y = y[keep]

y = np.expand_dims(y, 1)


#inspect data

#(plot digit)

# create a (binary cross entropy) loss function as discussed in the slides:
def avrLoss(y, a):  #page59 in week1's slides     Binary cross-entropy function
    eps = 1e-7
    a = np.maximum(a, eps)
    a = np.minimum(a, 1 - eps)
    loss = -(y * np.log(a) + (1 - y) * np.log(1 - a))
    return np.mean(loss)


#test loss function:
#(just run this cell and see what happens)


loss = avrLoss(np.array([0, 1, 0, 1]), np.linspace(.1, .9, 4))
print(loss)
assert np.all(loss == 0.5543313122608056)


#create a sigmoid          page 58
def sigmoid(z):
    z = np.maximum(z, -700)
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z = np.maximum(z, -700)
    softmax = np.exp(z) / np.sum(np.exp(z)) #TODO è sbaglatia di sicuro correggi
    return softmax


#check sigmoid
#(just run this cell and see what happens)

out = sigmoid(np.array(range(-5, 5)))
print(out)
#assert np.all(np.around(out, 8) == np.array(   [0.00669285, 0.01798621, 0.04742587, 0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708, 0.95257413, 0.98201379]))
#make test & train data

Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X, y, test_size=0.3, shuffle=True)
#train

