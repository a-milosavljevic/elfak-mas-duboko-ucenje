import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

print("Tensorflow version: " + tf.__version__)
print("Keras version: " + tf.keras.__version__)


########################################################################################################################
# PARAMETERS
########################################################################################################################

max_features = 10000  # number of words to consider as features
max_len = 500  # cut texts after this number of words (among top max_features most common words)
embedding_dim = 50
batch_size = 2048
init_lr = 1e-3


########################################################################################################################
# LOADING DATA
########################################################################################################################

print('Loading data...')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('x_test shape:', y_test.shape)
print('x_train sample:', x_train[0])
print('y_train samples:', y_train[:10])


########################################################################################################################
# BUILDING MODEL
########################################################################################################################

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input((max_len,)))
model.add(tf.keras.layers.Embedding(max_features, embedding_dim))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.binary_accuracy])

model.summary()


########################################################################################################################
# TRAINING MODEL
########################################################################################################################

path = os.path.join(os.getcwd(), 'trained_model.keras')
save_model = tf.keras.callbacks.ModelCheckpoint(path, monitor='val_binary_accuracy', mode='max',
                                                verbose=1, save_best_only=True)

csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(os.getcwd(), 'training.csv'))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max',
                                                  patience=30, restore_best_weights=True, verbose=1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_binary_accuracy', mode='max',
                                                 factor=0.1, patience=10, verbose=1)

hist = model.fit(x_train, y_train,
                 epochs=1000,
                 batch_size=batch_size,
                 validation_data=(x_test, y_test),
                 shuffle=True,
                 callbacks=[save_model,
                            csv_logger,
                            early_stopping,
                            reduce_lr],
                 verbose=2)

model.save(path, include_optimizer=False)


########################################################################################################################
# EVALUATE
########################################################################################################################

plt.clf()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()

plt.clf()
plt.plot(hist.history['binary_accuracy'])
plt.plot(hist.history['val_binary_accuracy'])
plt.show()

eval = model.evaluate(x_test, y_test)
print(eval)