import tensorflow as tf
import matplotlib.pyplot as plt
import os

print("Tensorflow version: " + tf.__version__)
print("Keras version: " + tf.keras.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer((28, 28, 1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='tanh'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.summary()

hist = model.fit(x_train, y_train, batch_size=2048, epochs=100, validation_data=(x_test, y_test), verbose=2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()
plt.clf()
plt.plot(hist.history['sparse_categorical_accuracy'])
plt.plot(hist.history['val_sparse_categorical_accuracy'])
plt.show()

res = model.evaluate(x_test, y_test, batch_size=2048)
print(res)