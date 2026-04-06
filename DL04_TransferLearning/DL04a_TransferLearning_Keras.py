import os
import urllib.request
import zipfile
import tarfile
import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt

print("Tensorflow verzija: " + tf.__version__)
print("Keras verzija: " + tf.keras.__version__)

########################################################################################################################
# 1. PODEŠAVANJE LOKALNIH PUTANJA ZA REZULTATE
########################################################################################################################

# Putanja u tekućem direktorijumu gde će se čuvati rezultati
local_results_folder = './train_results_keras'
if not os.path.exists(local_results_folder):
    os.makedirs(local_results_folder)
    print(f"Napravljen direktorijum za rezultate: {local_results_folder}")


########################################################################################################################
# 2. PREUZIMANJE I RASPAKIVANJE PODATAKA (LOKALNO)
########################################################################################################################

dataset_url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
zip_path = "./caltech-101.zip"
caltech_folder = "./caltech-101"
data_folder = "./101_ObjectCategories"

if not os.path.exists(zip_path):
    print("Preuzimanje dataseta...")
    urllib.request.urlretrieve(dataset_url, zip_path)
    print("Preuzimanje završeno.")

if not os.path.exists(caltech_folder):
    print("Raspakivanje osnovne ZIP arhive...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./')

if not os.path.exists(data_folder):
    print("Raspakivanje slika iz TAR.GZ arhive...")
    tar_path = os.path.join(caltech_folder, '101_ObjectCategories.tar.gz')
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall('./')
    print("Slike su spremne.")


########################################################################################################################
# 3. FUNKCIJE ZA UČITAVANJE PODATAKA I KREIRANJE MODELA
########################################################################################################################

def load_data(image_size, keep_aspect=True, fill_value=0):
    categories = []
    train_data = []
    test_data = []

    image_folders = sorted(os.listdir(data_folder))
    for c in image_folders:
        # Izbacivanje kategorije BACKGROUND_Google
        if c == 'BACKGROUND_Google':
            continue
            
        path = os.path.join(data_folder, c)
        if os.path.isdir(path):
            categories.append(c)

            cnt = 0
            files = sorted(os.listdir(path))
            for f in files:
                image_file = os.path.join(path, f)
                if os.path.isfile(image_file):
                    if cnt % 5 == 2:
                        test_data.append((image_file, categories.index(c)))
                    else:
                        train_data.append((image_file, categories.index(c)))
                    cnt += 1

    x_train = np.zeros((len(train_data), image_size, image_size, 3), np.uint8)
    y_train = np.zeros((len(train_data), 1), np.uint8)
    x_test = np.zeros((len(test_data), image_size, image_size, 3), np.uint8)
    y_test = np.zeros((len(test_data), 1), np.uint8)

    for i in range(len(train_data)):
        file, c = train_data[i]
        y_train[i] = c
        img = cv.imread(file)
        if img is None:
            continue
        if keep_aspect:
            cy = img.shape[0]
            cx = img.shape[1]
            if cx > cy:
                d = cx - cy
                img = cv.copyMakeBorder(img, top=d//2, bottom=d-d//2, left=0, right=0,
                                        borderType=cv.BORDER_CONSTANT, value=fill_value)
            else:
                d = cy - cx
                img = cv.copyMakeBorder(img, top=0, bottom=0, left=d//2, right=d-d//2,
                                        borderType=cv.BORDER_CONSTANT, value=fill_value)
        img = cv.resize(img, (image_size, image_size), interpolation=cv.INTER_LINEAR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x_train[i] = img

    for i in range(len(test_data)):
        file, c = test_data[i]
        y_test[i] = c
        img = cv.imread(file)
        if img is None:
            continue
        if keep_aspect:
            cy = img.shape[0]
            cx = img.shape[1]
            if cx > cy:
                d = cx - cy
                img = cv.copyMakeBorder(img, top=d // 2, bottom=d - d // 2, left=0, right=0,
                                        borderType=cv.BORDER_CONSTANT, value=fill_value)
            else:
                d = cy - cx
                img = cv.copyMakeBorder(img, top=0, bottom=0, left=d // 2, right=d - d // 2,
                                        borderType=cv.BORDER_CONSTANT, value=fill_value)
        img = cv.resize(img, (image_size, image_size), interpolation=cv.INTER_LINEAR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x_test[i] = img

    rand_train_idx = np.random.RandomState(seed=0).permutation(len(train_data))
    x_train = x_train[rand_train_idx]
    y_train = y_train[rand_train_idx]

    rand_test_idx = np.random.RandomState(seed=0).permutation(len(test_data))
    x_test = x_test[rand_test_idx]
    y_test = y_test[rand_test_idx]

    return categories, train_data, test_data, x_train, y_train, x_test, y_test


def create_model(x_train, x_test, classes, trainable_encoder=False):
    input_shape = x_train.shape[1:]

    x = tf.keras.layers.Input(shape=input_shape, name='input')

    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet',
                                                    input_tensor=x, pooling='avg', classes=classes)

    if not trainable_encoder:
        for layer in backbone.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    y = backbone.output
    y = tf.keras.layers.Dropout(rate=0.5)(y)
    y = tf.keras.layers.Dense(classes, activation='softmax', name='output')(y)

    model = tf.keras.models.Model(inputs=x, outputs=y)

    x_train_pp = tf.keras.applications.efficientnet.preprocess_input(x_train)
    x_test_pp = tf.keras.applications.efficientnet.preprocess_input(x_test)
    
    return x_train_pp, x_test_pp, model


########################################################################################################################
# 4. UČITAVANJE PODATAKA
########################################################################################################################

image_size = 224
batch_size = 32

print("Učitavanje i pretprocesiranje slika...")
categories, train_data, test_data, x_train, y_train, x_test, y_test = load_data(image_size, keep_aspect=True, fill_value=(127, 127, 127))
print("Oblik trening skupa:", x_train.shape, y_train.shape)
print("Oblik test skupa:", x_test.shape, y_test.shape)
print("Broj kategorija:", len(categories))


########################################################################################################################
# 5. KREIRANJE MODELA I CALLBACK-OVA
########################################################################################################################

x_train_pp, x_test_pp, model = create_model(x_train, x_test, classes=len(categories), trainable_encoder=False)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max',
                                                  patience=30, restore_best_weights=True, verbose=1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', mode='max',
                                                 factor=0.1, patience=10, verbose=1)

csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(local_results_folder, 'training.csv'))


########################################################################################################################
# 6. OBUČAVANJE MODELA
########################################################################################################################

print("Početak treniranja...")
hist = model.fit(x_train_pp, y_train, batch_size=batch_size, epochs=1000,
                 validation_data=(x_test_pp, y_test), verbose=1,
                 callbacks=[early_stopping, reduce_lr, csv_logger])

model_path = os.path.join(local_results_folder, 'trained_model.keras')
model.save(model_path, include_optimizer=False)
print(f"Model sačuvan na: {model_path}")


########################################################################################################################
# 7. ČUVANJE GRAFIKA LOKALNO
########################################################################################################################

plt.clf()
plt.plot(hist.history['loss'], label='Trening')
plt.plot(hist.history['val_loss'], label='Validacija')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(local_results_folder, 'training_loss.png'))
plt.close()

plt.clf()
plt.plot(hist.history['sparse_categorical_accuracy'], label='Trening')
plt.plot(hist.history['val_sparse_categorical_accuracy'], label='Validacija')
plt.legend()
plt.title('Accuracy')
plt.savefig(os.path.join(local_results_folder, 'training_accuracy.png'))
plt.close()


########################################################################################################################
# 8. EVALUACIJA MODELA
########################################################################################################################

print("\nEvaluacija modela...")
res_train = model.evaluate(x_train_pp, y_train, batch_size=batch_size, verbose=0)
res_test = model.evaluate(x_test_pp, y_test, batch_size=batch_size, verbose=0)
print(f"Trening rezultati (Loss, Accuracy): {res_train}")
print(f"Test rezultati (Loss, Accuracy): {res_test}")

y_out = model.predict(x_test_pp, batch_size=batch_size)
y_out = np.argmax(y_out, axis=1)

misclassified_folder = os.path.join(local_results_folder, 'misclassified')
if not os.path.exists(misclassified_folder):
    os.makedirs(misclassified_folder)

i = 0
y_test_flat = y_test.flatten()
for img, out, exp in zip(x_test, y_out, y_test_flat):
    if out != exp:
        i += 1
        title = '{} as {}'.format(categories[exp], categories[out])
        
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(title)
        plt.savefig(os.path.join(misclassified_folder, '{} ({}).jpg'.format(i, title)))
        plt.close()

print(f"\nPogrešne predikcije ({i} komada) su sačuvane u lokalnom folderu: {misclassified_folder}")