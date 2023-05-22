import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# path
path = os.path.join(os.path.dirname(__file__), 'img1.png')

# captura
image = cv2.imread(path)

# alto, ancho, canal
alto, ancho, canal = image.shape
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# preprocesamiento
if canal == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(1, 2, 1)
ax = plt.imshow(image, cmap='gray')
ax.axes.set_axis_off()

# Definir el umbral de área
area_threshold = 250
(umbral, binary_img) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Aplicar la operación de apertura para eliminar los objetos pequeños
opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

# Encontrar los contornos de los objetos después de la operación morfológica
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear una máscara para los objetos que cumplen el umbral de área
mask = np.zeros_like(binary_img)
for contour in contours:
    area = cv2.contourArea(contour)
    if area >= area_threshold:
        cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)

# Aplicar la máscara a la imagen original
result = cv2.bitwise_and(binary_img, mask)

plt.subplot(1, 2, 2)
ax = plt.imshow(result, cmap='gray')
ax.axes.set_axis_off()


# mostrar imagen
plt.show()
