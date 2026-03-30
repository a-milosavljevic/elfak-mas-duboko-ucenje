import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

print("Tensorflow version: " + tf.__version__)
print("Keras version: " + tf.keras.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0

# PCA - priprema podataka i određivanje broja komponenti
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

pca = PCA(n_components=500)
pca.fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# PCA - odredjivanje transformacije i transformacija trening i test ulaznih podataka
pca = PCA(n_components=100) # broj komonenenti izabran na bazi prethodnog grafikona
x_pca_train = pca.fit_transform(x_train)
x_pca_test = pca.transform(x_test)
pca_std = np.std(x_pca_train)
print(x_pca_train.shape, x_pca_test.shape)

# PCA - validacija (inverzna transformacija)
inv_pca = pca.inverse_transform(x_pca_test)

# Uporedni prikaz originalnih i rekonstruisanih podataka
def side_by_side(indexes):
    org = x_test[indexes].reshape(28, 28)
    rec = inv_pca[indexes].reshape(28, 28)
    pair = np.concatenate((org, rec), axis=1)
    plt.figure(figsize=(4, 2))
    plt.imshow(pair)
    plt.show()

# Uporedni prikaz 10 slika u cilju validacije broja komponenti
for index in range(0, 10):
    side_by_side(index)

# Kreiranje MLP modela
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer((x_pca_train.shape[1],)))
model.add(tf.keras.layers.Dense(256, activation='tanh'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.summary()

# Treniranje modela
hist = model.fit(x_pca_train, y_train, batch_size=2048, epochs=500, validation_data=(x_pca_test, y_test), verbose=2)

# Evaluacija i prikaz grafikona
res = model.evaluate(x_pca_test, y_test, batch_size=256)
print(res)

plt.clf()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()

plt.clf()
plt.plot(hist.history['sparse_categorical_accuracy'])
plt.plot(hist.history['val_sparse_categorical_accuracy'])
plt.show()