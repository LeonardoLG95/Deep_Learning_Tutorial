# Transfer learning se trata de descargar modelos ya creados y entrenados para usar
#   se pueden encontrar en tensorflow hub (y posiblemente en otros sitios)
import tensorflow as tf
import matplotlib.pylab as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import numpy as np
import PIL.Image as Image

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Descargar el modelo mobilenet, este modelo espera imagenes de 224 x 224 px
CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_RES = 224

model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
])

# El modelo esta pensado para tener 1000 outputs diferentes (cosas que puede detectar)
#   y entre ellas estan los uniformes militares, como esta imagen que vamos a descargar
grace_hopper = tf.keras.utils.get_file('image.jpg',
                                       'https://storage.googleapis.com/download.tensorflow.org/'
                                       'example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))
grace_hopper = np.array(grace_hopper)/255.0
result = model.predict(grace_hopper[np.newaxis, ...])

predicted_class = np.argmax(result[0], axis=-1)
