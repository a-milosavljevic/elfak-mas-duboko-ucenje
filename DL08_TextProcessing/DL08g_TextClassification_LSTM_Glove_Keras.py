import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import urllib.request
import zipfile

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
# DOWNLOADING GLOVE 6B EMBEDDINGS
########################################################################################################################

# Download and unpack into project root
glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
zip_path = "./glove.6B.zip"
glove_path = f'./glove.6B.{embedding_dim}d.txt'

if not os.path.exists(zip_path):
    print("Downloading Glove 6B embeddings...")
    urllib.request.urlretrieve(glove_url, zip_path)
    print("Finished.")

if not os.path.exists(glove_path):
    print("Unpacking embeddings...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./')


########################################################################################################################
# LOADING EMBEDDINGS
########################################################################################################################

embeddings_index = {}
f = open(glove_path, encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((max_features, embedding_dim))
word_index = tf.keras.datasets.imdb.get_word_index()
print("Found %s word indexes in IMDB set." % len(word_index))

cnt_skiped = 0
cnt_processed = 0
index_offset = 3 # Keras offset za IMDB zbog dodavanja specijalnih tokena <PAD>, <START> i <UNK>

for word, i in word_index.items():
    adjusted_index = i + index_offset # Usklađujemo indeks

    if adjusted_index < max_features:
        embedding_vector = embeddings_index.get(word)
        cnt_processed += 1
        if embedding_vector is not None:
            # Reči pronađene u embedding indexu dobijaju svoje vektore
            embedding_matrix[adjusted_index] = embedding_vector
    else:
        cnt_skiped += 1

print(cnt_processed, cnt_skiped)


########################################################################################################################
# BUILDING MODEL
########################################################################################################################

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input((max_len,)))
model.add(tf.keras.layers.Embedding(max_features, embedding_dim))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Setting embedding matrix
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

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
                                                  patience=20, restore_best_weights=True, verbose=1)

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
# EVALUATE TRAINED MODEL
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


########################################################################################################################
# FINE-TUNING MODEL
########################################################################################################################

# Unlocking embedding layer
model.layers[0].trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1*init_lr)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.binary_accuracy])

path = os.path.join(os.getcwd(), 'finetuned_model.keras')
save_model = tf.keras.callbacks.ModelCheckpoint(path, monitor='val_binary_accuracy', mode='max',
                                                verbose=1, save_best_only=True)

csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(os.getcwd(), 'finetuning.csv'))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max',
                                                  patience=20, restore_best_weights=True, verbose=1)

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
# EVALUATE FINE-TUNED MODEL
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