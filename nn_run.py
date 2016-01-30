from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution1D
import numpy as np

X_train = np.load("data/cd00001.smp.npy")/1000
Y_train = X_train[:,7,:]

X_test = np.load("data/cd00002.smp.npy")/1000
Y_test = X_test[:,7,:]

model = Sequential()

model.add(Convolution1D(64, 5, input_dim=20, init='glorot_normal', input_length=15, border_mode='same', activation='relu'))
model.add(Dropout(0.5))

model.add(Convolution1D(64, 5, init='glorot_normal', border_mode='same', activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())
# model.add(Flatten(input_shape=(15,20)))

model.add(Dense(20, init='glorot_normal', activation='tanh'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(
    X_train, 
    Y_train, 
    nb_epoch=2000, 
    batch_size=5, 
    validation_data=(X_test, Y_test),
    show_accuracy=True
)

score = model.predict(X_test, batch_size=16)
