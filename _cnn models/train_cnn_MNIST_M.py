#! /usr/bin/env python

"""Code adapted from https://towardsdatascience.com/a-quick-introduction-to-tensorflow-2-0-for-deep-learning-e740ca2e974c"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
import h5py

model_file = 'model_MNIST_M.h5' # output

hf = h5py.File('MNIST_M_dataset.h5', 'r')
train_images = hf['train_images'][:]
train_labels = hf['train_labels'][:]
test_images = hf['test_images'][:]
test_labels = hf['test_labels'][:]
hf.close()

IMG_SIZE = (28, 28, 1)
input_img = layers.Input(shape=IMG_SIZE)

model = layers.Conv2D(32, (3, 3), padding='same')(input_img)
model = layers.Activation('relu')(model)

model = layers.Conv2D(32, (3, 3), padding='same', strides=(2, 2))(model)
model = layers.Activation('relu')(model)

model = layers.Conv2D(64, (3, 3), padding='same')(model)
model = layers.Activation('relu')(model)

model = layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2))(model)
model = layers.Activation('relu')(model)

model = layers.Conv2D(64, (3, 3), padding='same')(model)
model = layers.Activation('relu')(model)

model = layers.Conv2D(64, (3, 3), padding='same')(model)
model = layers.Activation('relu')(model)

model = layers.GlobalAveragePooling2D()(model)

model = layers.Dense(32)(model)
model = layers.Activation('relu')(model)

model = layers.Dense(11)(model)
output_img = layers.Activation('softmax')(model)

model = models.Model(input_img, output_img)

model.summary()

train_images = train_images.reshape(66000, 28, 28, 1).astype('float32') / 255.0
test_images = test_images.reshape(11000, 28, 28, 1).astype('float32') / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels, 11)
test_labels = tf.keras.utils.to_categorical(test_labels, 11)

adam = optimizers.Adam(lr=0.0001)
model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])

history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.7, 1])
plt.legend(loc='best')

test_loss, test_accuracy = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy = {0:.2f}%'.format(test_accuracy*100.0))

model.save(model_file)