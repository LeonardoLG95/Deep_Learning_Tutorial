import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Feature: datos de entrada (inputs) en este caso celsius
# Labels: datos de salida (outputs) Fahrenheit
# Example: Un par de datos, uno de entrada y uno de salida. Usados para el entrenamiento

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius):
    print(f'{c} degrees Celsius = {fahrenheit[i]} degrees Fahrenheit.')

# Crear el modelo:
# input_shape: las dimensiones de array que queremos tener, en este caso 1,
#       si por ejemplo fuese una imagen serian 2 'x' y 'z'
# units: la cantidad de variables que tiene que aprender en caso de ser capas internas
#       debere preveer cuantas han de ser, en este caso como es la unica y ultima es la variable de output (fahrenheit)
layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# Cuando las capas estan definidas, en un modelo secuencial, deben ir en un array
model = tf.keras.Sequential([layer_0])

'''
Tambien se podrian ver asi:
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
'''

# optimizer: el tipo de funcion que se usara para
#           realizar ajustes a las variables internas en cada iteracion
# loss: utilizada al entrenar el modelo, para ajustar los pesos en las iteraciones al
#           intentar que un resultado aleatorio se acerque al deseado (recordar que al entrenar se meten valores de salida)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(1))

# fit: entrenar al modelo
#       primero los inputs, segundo los output deseados
# epoch: la cantidad de veces que el ciclo se repetira
history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)
print("Finished training the model")

# matplotlib.use("TkAgg")
# fit devuelve el historial de perdida durante cada iteracion
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
print(history.history['loss'])
print(f"Numero de datos en el historia : {len(history.history['loss'])}")

# Ejecutar una prediccion
print(model.predict([100.0]))
print("These are the layer variables: {}".format(layer_0.get_weights()))
plt.show()
