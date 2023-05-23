# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:30:47 2023

@author: tvo
"""

import cv2
from matplotlib import pyplot as plt

def deteccionBordes(imagen, umbral):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar umbralización binaria
    _, imagen_binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)

    return imagen_binaria

# Cargar la imagen
imagen_original = cv2.imread('C:/Users/tvo/Downloads/Manzana.jpg')

# Especificar el umbral (0-255)
umbral = 128

# Aplicar umbralización binaria
imagen_binaria = deteccionBordes(imagen_original, umbral)

# Aplicar detección de bordes utilizando Canny
imagen_bordes = cv2.Canny(imagen_binaria, 100, 200)  # Ajusta los valores de umbral según sea necesario

# Mostrar la imagen binaria y la imagen de bordes
plt.subplot(1, 2, 1), plt.imshow(imagen_binaria, cmap='gray')
plt.title('Imagen Binaria'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(imagen_bordes, cmap='gray')
plt.title('Imagen de Bordes'), plt.xticks([]), plt.yticks([])

plt.show()
