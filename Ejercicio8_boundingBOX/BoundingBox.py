# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:35:57 2023

@author: tvo
"""

import cv2
import numpy as np

def dibujar_bounding_box_centroide(imagen, contornos):
    for contorno in contornos:
        # Obtener el cuadro delimitador del contorno
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Dibujar el cuadro delimitador en la imagen
        cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Calcular el centroide del contorno
        M = cv2.moments(contorno)
        if M["m00"] != 0:  # Verificar que el área del contorno no sea cero
            centroide_x = int(M["m10"] / M["m00"])
            centroide_y = int(M["m01"] / M["m00"])
            
            # Dibujar el centroide en la imagen
            cv2.circle(imagen, (centroide_x, centroide_y), 4, (0, 0, 255), -1)
    
    return imagen

# Cargar la imagen
imagen_original = cv2.imread('C:/Users/tvo/Downloads/herramientas.jpg')

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)

# Aplicar umbralización
_, imagen_umbralizada = cv2.threshold(imagen_gris, 128, 255, cv2.THRESH_BINARY)

# Buscar contornos en la imagen umbralizada
contornos, _ = cv2.findContours(imagen_umbralizada.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los cuadros delimitadores y centroides en la imagen original
imagen_resultado = dibujar_bounding_box_centroide(imagen_original.copy(), contornos)

# Mostrar la imagen con los cuadros delimitadores y centroides
cv2.imshow('Objetos detectados', imagen_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
