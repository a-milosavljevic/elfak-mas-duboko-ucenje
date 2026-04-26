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
# 3. DATA AUGMENTATION
########################################################################################################################

def get_data_generator():
    return tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=45,
                                                           width_shift_range=0.2,
                                                           height_shift_range=0.2,
                                                           brightness_range=(0.5, 1.5),
                                                           shear_range=0.1,
                                                           zoom_range=0.2,
                                                           fill_mode='constant', #'constant', 'nearest', 'reflect'
                                                           cval=0,
                                                           horizontal_flip=True,
                                                           vertical_flip=False,
                                                           validation_split=0.0,
                                                           dtype=np.float32)

# Test data augmentation
img = cv.imread(os.path.join(data_folder, 'butterfly', 'image_0018.jpg'))
img = cv.resize(img, (224, 224))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.clf()
plt.figure(figsize=(15, 9))  # Širina i visina prozora za prikaz

data_gen = get_data_generator()
for i in range(15):
    if i == 0:
        img_da = img
    else:
        img_da = data_gen.random_transform(img)
        img_da = np.clip(img_da, 0, 255).astype(np.uint8)
    plt.subplot(3, 5, i + 1)  # Kreiranje grida: 3 reda, 5 kolona, i-ta pozicija
    plt.imshow(img_da)
    plt.axis('off')  # Isključivanje x i y osa radi preglednijeg prikaza slika

plt.tight_layout()  # Automatsko podešavanje razmaka između slika
plt.savefig(os.path.join(local_results_folder, 'data_augmentation_test.png'))


########################################################################################################################
# 4. PODRŠKA ZA UČITAVANJE I AUGMENTACIJU PODATAKA
########################################################################################################################

def load_data_def():
    categories = []
    train_images = []
    test_images = []

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
                        test_images.append((image_file, categories.index(c)))
                    else:
                        train_images.append((image_file, categories.index(c)))
                    cnt += 1

    train_images = np.random.RandomState(0).permutation(train_images)
    test_images = np.random.RandomState(0).permutation(test_images)

    return categories, train_images, test_images

class DataGenerator(tf.keras.utils.Sequence):
    """ Klasa za učitavanje i augmentaciju slika u batch-eve """
    def __init__(self, batch_size, images, image_size, data_aug=False, keep_aspect=True, fill_value=0, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.images = images
        self.image_size = image_size
        self.keep_aspect = keep_aspect
        self.fill_value = fill_value

        self.data_aug = data_aug
        self.data_gen = None
        if data_aug:
           self.data_gen = get_data_generator()

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min(len(self.images), (idx + 1) * self.batch_size)
        batch_images = self.images[batch_start:batch_end]

        batch_x = np.zeros((len(batch_images), self.image_size, self.image_size, 3), dtype=np.float32)
        batch_y = np.zeros((len(batch_images),), dtype=np.int32)

        # Load and resize image
        for i in range(len(batch_images)):
            image_path, image_class = batch_images[i]
            img = cv.imread(image_path)
            if self.keep_aspect:
                cy = img.shape[0]
                cx = img.shape[1]
                if cx > cy:
                    d = cx - cy
                    img = cv.copyMakeBorder(img, top=d // 2, bottom=d - d // 2, left=0, right=0,
                                            borderType=cv.BORDER_CONSTANT, value=self.fill_value)
                else:
                    d = cy - cx
                    img = cv.copyMakeBorder(img, top=0, bottom=0, left=d // 2, right=d - d // 2,
                                            borderType=cv.BORDER_CONSTANT, value=self.fill_value)
            img = cv.resize(img, (self.image_size, self.image_size), interpolation=cv.INTER_NEAREST)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            if self.data_aug:
                img = self.data_gen.random_transform(img)

            batch_x[i] = img
            batch_y[i] = image_class

        return batch_x, batch_y


########################################################################################################################
# 5. FUNKCIJA ZA KREIRANJE MODELA
########################################################################################################################

def create_model(img_size, classes, trainable_encoder=False):
    x = tf.keras.layers.Input(shape=(img_size, img_size, 3), name='input')

    # Ugradjivanje predprocesiranja u model preko Lambda sloja
    x = tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input)(x)

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

    return model


########################################################################################################################
# 6. UČITAVANJE PODATAKA
########################################################################################################################

image_size = 224
batch_size = 32

print("Učitavanje i pretprocesiranje slika...")
categories, train_images, test_images = load_data_def()
print("Broj trening slika:", len(train_images))
print("Broj test slika:", len(test_images))
print("Broj kategorija:", len(categories))

datagen_train = DataGenerator(batch_size=batch_size,
                              images=train_images,
                              image_size=image_size,
                              data_aug=True,
                              keep_aspect=True)

datagen_test = DataGenerator(batch_size=batch_size,
                             images=test_images,
                             image_size=image_size,
                             data_aug=False,
                             keep_aspect=True)


########################################################################################################################
# 7. KREIRANJE MODELA
########################################################################################################################

model = create_model(image_size, classes=len(categories), trainable_encoder=False)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.summary()


########################################################################################################################
# 8. INICIJALNO OBUČAVANJE MODELA (WARMUP TRAINING)
########################################################################################################################

# Logovanje CSV-a na Google Drive
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(local_results_folder, 'warmup.csv'))

data_gen = get_data_generator()

print("Warmup training for 20 epochs...")
hist = model.fit(datagen_train,
                 epochs=20,
                 validation_data=datagen_test,
                 verbose=1,
                 callbacks=[csv_logger])

# Čuvanje modela na Google Drive
model_path = os.path.join(local_results_folder, 'warmup_model.keras')
model.save(model_path, include_optimizer=False)
print(f"Model sačuvan na: {model_path}")

# Prikaz i čuvanje grafikona
plt.clf()
plt.plot(hist.history['loss'], label='Trening')
plt.plot(hist.history['val_loss'], label='Validacija')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(local_results_folder, 'warmup_loss.png'))

plt.clf()
plt.plot(hist.history['sparse_categorical_accuracy'], label='Trening')
plt.plot(hist.history['val_sparse_categorical_accuracy'], label='Validacija')
plt.legend()
plt.title('Accuracy')
plt.savefig(os.path.join(local_results_folder, 'warmup_accuracy.png'))


########################################################################################################################
# 9. FINO OBUČAVANJE MODELA (FINE-TUNING)
########################################################################################################################

# Otključavanje svih slojeva
for layer in model.layers:
    layer.trainable = True

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max',
                                                  patience=20, restore_best_weights=True, verbose=1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', mode='max',
                                                 factor=0.1, patience=10, verbose=1)

# Logovanje CSV-a na Google Drive
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(local_results_folder, 'finetuning.csv'))

# Fine-tuning model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

print("Fine-tuning model...")
hist = model.fit(datagen_train,
                 epochs=1000,
                 validation_data=datagen_test,
                 verbose=1,
                 callbacks=[early_stopping, reduce_lr, csv_logger])

# Čuvanje modela na Google Drive
model_path = os.path.join(local_results_folder, 'finetuned_model.keras')
model.save(model_path, include_optimizer=False)
print(f"Model sačuvan na: {model_path}")

# Prikaz i čuvanje grafikona
plt.clf()
plt.plot(hist.history['loss'], label='Trening')
plt.plot(hist.history['val_loss'], label='Validacija')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(local_results_folder, 'finetuning_loss.png'))

plt.clf()
plt.plot(hist.history['sparse_categorical_accuracy'], label='Trening')
plt.plot(hist.history['val_sparse_categorical_accuracy'], label='Validacija')
plt.legend()
plt.title('Accuracy')
plt.savefig(os.path.join(local_results_folder, 'finetuning_accuracy.png'))


########################################################################################################################
# 10. EVALUACIJA MODELA
########################################################################################################################

print("\nEvaluacija modela...")
datagen_train_eval = DataGenerator(batch_size=batch_size,
                                   images=train_images,
                                   image_size=image_size,
                                   data_aug=False,
                                   keep_aspect=True)
res_train = model.evaluate(datagen_train_eval, batch_size=batch_size, verbose=0)
res_test = model.evaluate(datagen_test, batch_size=batch_size, verbose=0)
print(f"Trening rezultati (Loss, Accuracy): {res_train}")
print(f"Test rezultati (Loss, Accuracy): {res_test}")

# Izvlači sve labele za test podatke
test_labels = []
for i in range(len(datagen_test)):
    batch_x, batch_y = datagen_test[i]
    test_labels.extend(batch_y)
y_test = np.array(test_labels)

y_out = model.predict(datagen_test, batch_size=batch_size)
y_out = np.argmax(y_out, axis=1)

# Kreiranje podfoldera za pogrešne predikcije
misclassified_folder = os.path.join(local_results_folder, 'misclassified')
if not os.path.exists(misclassified_folder):
    os.makedirs(misclassified_folder)

# Iteriramo kroz ukupan broj test primera
i = 0
for idx in range(len(y_test)):
    out = y_out[idx]
    exp = y_test[idx]

    if out != exp:
        i += 1
        title = '{} as {}'.format(categories[int(exp)], categories[int(out)])

        batch_idx = idx // batch_size   # računanje indeksa batch-a
        idx_in_batch = idx % batch_size # računanje indeksa slike u batch-u

        batch_x, _ = datagen_test[batch_idx]
        img = np.clip(batch_x[idx_in_batch], 0, 255).astype(np.uint8)

        # Čuvanje svake pogrešne slike na Drive
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(title)
        plt.savefig(os.path.join(misclassified_folder, '{} ({}).jpg'.format(i, title)))
        plt.close()

print(f"\nPogrešne predikcije ({i} komada) su sačuvane na Drive-u u folderu: {misclassified_folder}")