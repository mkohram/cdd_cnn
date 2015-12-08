from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution1D
import numpy as np

X_train = np.load("data/cd00001.smp.npy")
Y_train = np.random.rand(X_train.shape[0], 20)
Y_train = Y_train/np.sum(Y_train,axis=1)[:,None]

X_test = np.load("data/cd00001.smp.npy")
Y_test = np.random.rand(X_train.shape[0], 20)
Y_test = Y_train/np.sum(Y_train,axis=1)[:,None]

model = Sequential()

model.add(Convolution1D(64, 5, input_dim=20, init='uniform', input_length=15, border_mode='same'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(20, init='uniform', activation='relu'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, Y_train, nb_epoch=20, batch_size=16)
score = model.predict(X_test, batch_size=16)
