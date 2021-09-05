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

# Descargar el modelo mobilenet
CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_RES = 224

model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
])
