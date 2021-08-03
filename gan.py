from numpy import ones
from numpy import vstack
from data import generateFakeSamples, generateLatentPoints, generateRealSamples, loadRealSamples
from generator import createGenerator
from discriminator import createDiscriminator
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def createGan(discriminator: Sequential, generator: Sequential):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def train(discriminator:Sequential, generator:Sequential, gan:Sequential, dataset, latentDim, numEpochs=100, numBatch=256, evalFreq=10):
    batchPerEpoch = int(dataset.shape[0] / numBatch)
    halfBatch = int(numBatch / 2)
    for i in range(numEpochs):
        for j in range(batchPerEpoch):
            xReal, yReal = generateRealSamples(dataset, halfBatch)
            # discriminator.train_on_batch(xReal, yReal)

            xFake, yFake = generateFakeSamples(generator, latentDim, halfBatch)
            # discriminator.train_on_batch(xFake, yFake)
            
            X, y = vstack((xReal, xFake)), vstack((yReal, yFake))
            dLoss, _ = discriminator.train_on_batch(X, y)
        
            xGan = generateLatentPoints(latentDim, numBatch)
            yGan = ones((numBatch, 1))
            gLoss = gan.train_on_batch(xGan, yGan)

            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, batchPerEpoch, dLoss, gLoss))

        if (i+1) % evalFreq == 0:
            evaluatePerformance(i, discriminator, generator, dataset, latentDim)

def evaluatePerformance(epoch, discriminator: Sequential, generator: Sequential, dataset, latentDim, numSamples=100):
    xReal, yReal = generateRealSamples(dataset, numSamples)
    _, accReal = discriminator.evaluate(xReal, yReal, verbose=0)

    xFake, yFake = generateFakeSamples(generator, latentDim, numSamples)
    _, accFake = discriminator.evaluate(xFake, yFake, verbose=0)

    print(epoch, accReal, accFake)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (accReal*100, accFake*100))

    savePlot(xFake, epoch)

    filename = 'eval/generated_model%03d.h5' % (epoch + 1)
    generator.save(filename)

def savePlot(examples, epoch, n=10):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')

    filename = 'eval/generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()

if __name__ == '__main__':
    discriminator = createDiscriminator()
    generator = createGenerator()
    gan = createGan(discriminator, generator)
    gan.summary()