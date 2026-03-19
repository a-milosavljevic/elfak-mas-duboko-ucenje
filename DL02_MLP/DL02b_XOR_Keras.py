import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

print("Tensorflow version: " + tf.__version__)
print("Keras version: " + tf.keras.__version__)

data = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])
inputs = data[:, :2]
#inputs[inputs == 0] = -1
outputs = data[:, 2:]

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer((2,)),
    tf.keras.layers.Dense(5, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.0),
              loss=tf.keras.losses.mse,
              metrics=[tf.keras.metrics.binary_accuracy])

model.summary()

hist = model.fit(inputs, outputs, batch_size=4, epochs=1000, verbose=2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['binary_accuracy'])
plt.show()

res = model.evaluate(inputs, outputs, batch_size=4)
print(res)

out = model.predict(inputs)
for x, t, y in zip(inputs, outputs, out):
    print(x, '->', t, '=>', y)


