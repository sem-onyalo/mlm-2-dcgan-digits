from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def createDiscriminator(n_inputs=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), (2,2), padding='same', input_shape=n_inputs))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), (2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    discriminator = createDiscriminator()
    discriminator.summary()