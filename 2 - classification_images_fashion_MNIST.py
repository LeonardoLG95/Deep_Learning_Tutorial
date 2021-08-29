import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# dividir datasets coger el de entrenamiento y el de test
# metadata contiene los labels
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Extraer los nombres de los tipos de ropa de los metadatos del dataset
class_names = metadata.features['label'].names
print(f'Class names : {class_names}')

# Asegurarse de que cada muestra tiene el numero de ejemplos deseados
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print(f"Number of training examples : {num_train_examples}")
print(f"Number of test examples: {num_test_examples}")


# Para que el modelo funcione adecuadamente la informacion de la imagen
# debe de estar normalizada, los pixeles son un int de [0, 255] y necesitan
# estar en un rango de [0, 1]


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# Map aplica la funcion normalizar a cada elemento de los datasets


train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Usar cache hara que las mantenga en memoria acelerando el entrenamiento
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# # Quitar el color de la imagen
# for image, label in test_dataset.take(1):
#     break
# image = image.numpy().reshape((28, 28))
#
# # Muestra una de las imagenes
# plt.figure()
# plt.imshow(image, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# # Verificar las 25 primeras images del set de entrenamiento
# plt.figure(figsize=(10, 10))
# for i, (image, label) in enumerate(train_dataset.take(25)):
#     image = image.numpy().reshape((28, 28))
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image, cmap=plt.cm.binary)
#     plt.xlabel(class_names[label])
# plt.show()

# Creando el modelo y las capas:
#   la primera es de input, transforma las imagenes (array de 2D de 28x28)
#       a un array de 1D this esta solo cambia el formato, no realiza nada
#   la segunda es oculta, las 128 neuronas cogen los 784 pixeles
#       (tambien llamados nodos, del resultado de 28x28) y expulsa un unico valor
#   la tercera es de output, son 10 neuronas por los posibles 10 resultados que nos puede dar
#       (los tipos de ropa que hay) estos cogen el resultado de las 128 neuronas de la capa anterior
#       expulsara un valor entre 0 y 1 para cada tipo de ropa siendo la suma total 1
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compilar el modelo:
#   Optimizer, algoritmo para ajustar los parametros internos para minimizar la perdida
#   Loss function, un algoritmo para medir como de lejos esta el output del modelo del output deseado
#   Metrics, usado para monitorizar el entrenamiento, en este caso el modelo usa precision para
#       medir de cual de los 10 tipos de ropa puede ser la images
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

BATCH_SIZE = 32
# repeat hace que el entrenamiento se repita de forma infinita (limitado por las epoch)
# shuffle mezcla las muestras para que el algoritmo no pueda aprender el orden
# batch son las cantidades de imagenes y etiquetas para actualizar el modelo a cada pasada
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

# Entrenamiento
model.fit(train_dataset, epochs=30, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE))

# Comparar como el modelo funciona con el dataset de prueba, la precision sera menor que en el entrenamiento
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples / BATCH_SIZE))
print('Accuracy on test dataset:', test_accuracy)

# Predicciones
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

print(predictions.shape)

# Para ver la primera prediccion
print(predictions[0])

# Una prediccion esta compuesta de 10 numeros, los cuales representan un porcentaje de certeza de que una
#   prenda pertenece a uno de los 10 tipos
print(f"Prediccion de tipo : {np.argmax(predictions[0])}")
print(f"Tipo real : {test_labels[0]}")


def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()
# Plot las primeras X imagenes de test, la verdadera etiqueta, y la etiqueta predicha
# Predicciones correctas son azules, equivocadas rojas
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# Predecir sobre una sola imagen
img = test_images[0]
print(img.shape)

# Agregar la imagen al lote (batch)
img = np.array([img])
print(img.shape)

prediction_single = model.predict(img)
print(prediction_single)

plot_value_array(0, prediction_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
np.argmax(prediction_single[0])
