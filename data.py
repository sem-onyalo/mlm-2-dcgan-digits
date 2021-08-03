from keras.backend import expand_dims
from keras.datasets.mnist import load_data
from keras.models import Sequential
from numpy.random import randint
from numpy.random import randn
from numpy import zeros
from numpy import ones

def loadRealSamples():
    # load mnist dataset
    (trainX, _), (_, _) = load_data()
    # expand to 3d, i.e. add channel dimension
    X = expand_dims(trainX, axis=-1)
    # convert from unsigned ints to floats
    X = X.numpy().astype('float32')
    # convert scale from 0,255 to 0,1 (best practise is -1,1)
    X = X / 255.0
    return X

def generateRealSamples(dataset, numSamples):
    # choose random instances
    ix = randint(0, dataset.shape[0], numSamples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels
    y = ones((numSamples, 1))
    return X, y

def generateLatentPoints(latentDim, numSamples):
    xInput = randn(latentDim * numSamples)
    xInput = xInput.reshape((numSamples, latentDim))
    return xInput

def generateFakeSamples(generator: Sequential, latentDim, numSamples):
    xInput = generateLatentPoints(latentDim, numSamples)
    # predict outputs
    X = generator.predict(xInput)
    # create 'fake' class labels
    y = zeros((numSamples, 1))
    return X, y