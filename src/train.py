from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import layers
from keras import models

train = pd.read_csv("../data/train.csv")
test_images = (pd.read_csv("../data/test.csv").values).astype('float32')

train_images = (train.ix[:, 1:].values).astype('float32')
train_labels = train['label'].values.astype('int32')
train_images = train_images.reshape((42000,28,28,1))
train_labels = to_categorical(train_labels)

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.33, random_state=42)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1, batch_size=128)
model.evaluate(X_test,y_test)