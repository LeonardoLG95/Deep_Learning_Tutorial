import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt
import numpy as np

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Un dataset que proviene de Microsoft Research, pero que se almacena en google,
#   se extrae con una funcion de tensorflow
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)

# Set los paths a las distintas colecciones
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Ver la cantidad de datos que tenemos para cada uno:
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('Total imagenes de entrenamiento de gatos : ', num_cats_tr)
print('Total imagenes de entrenamiento de perros : ', num_dogs_tr)

print('Total imagenes de validacion de gatos : ', num_cats_val)
print('Total imagenes de validacion de perros : ', num_dogs_val)
print("--")
print("Total de imagenes de entrenamiento : ", total_train)
print("Total de imagenes de validacion : ", total_val)

BATCH_SIZE = 100  # Numero de ejemplos que se le pasar al entrenamiento antes de actualizar los pesos
IMG_SHAPE = 150  # Tamano de la imagen 150 x 150 px

# Las imagenes deben de estar preprocesadas apropiadamente para introducirse en la red.
# Los pasos serian:
#   Leer las imagenes del disco
#   Decodificar las imagenes y convertirlas en una cuadricula por contenido RGB
#   Convertirlas en floating point tensors
#   Reescalar los tensors de valores de 0 a 255 a entre 0 y 1

# todos estos pasos se realizan con la funcion ImageDataGenerator
train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

# Estas lineas aplican todos los cambios que realizan los generators
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),  # (150 x 150 px)
                                                           class_mode='binary')  # La forma de clasificar, en este caso perros o gatos (binario)

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE),
                                                              class_mode='binary')
# Para ver las imagenes:
sample_training_images, _ = next(train_data_gen)


# Funcion para mostrarlas en cuadriculas
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


plot_images(sample_training_images[:5])

# Entrenar el modelo:
# En principio son 3 por los 3 colores y una por la union,
#   despues una Dense para decidir cual es
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Muestra las variables del modelo:
model.summary()

# Entrenar el modelo:
EPOCHS = 20
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

# Visualizar los resultados del entrenamiento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./4-cnn_color_images.jpg')
plt.show()
