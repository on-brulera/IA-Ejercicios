# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:23:51 2023

@author: tvo
"""

import cv2

def umbralizacion_binaria(imagen, umbral):
    # Convertir la imagen a escala de grises si es necesario
    if len(imagen.shape) > 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbralización binaria
    _, imagen_binaria = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)

    return imagen_binaria

# Cargar la imagen
imagen_original = cv2.imread('C:/Users/tvo/Downloads/carro.jpg')

# Especificar el umbral (0-255)
umbral = 128

# Aplicar umbralización binaria
imagen_binaria = umbralizacion_binaria(imagen_original, umbral)

#Imagen original
cv2.imshow('Imagen Original',imagen_original)

# Mostrar la imagen binaria
cv2.imshow('Imagen Binaria', imagen_binaria)
cv2.waitKey(0)
cv2.destroyAllWindows()