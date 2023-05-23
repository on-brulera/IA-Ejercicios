import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# path
path = os.path.join(os.path.dirname(__file__), 'lineas.jpg')

# captura
image = cv2.imread(path)

# preprocesamiento
row, column, canal = image.shape

if canal == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Transformar la imagen a binaria
(umbral, binary_img) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Aplicar la operación de apertura
horizontal_size = column // 10
vertical_size = row // 10

# Lineas horizontales
shape = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 3))
horizontal = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, shape)
horizontal_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
horizontal_num = len(horizontal_contours)

# Lineas verticales
shape = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vertical_size))
vertical = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, shape)
vertical_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vertical_num = len(vertical_contours)


# Dibujar
plt.subplot(1, 3, 1)
plt.title('Imagen binaria')
ax = plt.imshow(binary_img, cmap='gray')
ax.axes.set_axis_off()

# Dibujar lineas horizontales
plt.subplot(1, 3, 2)
plt.title(f'Horizontales: {horizontal_num}')
ax = plt.imshow(horizontal, cmap='gray')
ax.axes.set_axis_off()

# Dibujar lineas verticales
plt.subplot(1, 3, 3)
plt.title(f'Herticales: {vertical_num}')
ax = plt.imshow(vertical, cmap='gray')
ax.axes.set_axis_off()


# Mostrar la imagen
plt.show()
