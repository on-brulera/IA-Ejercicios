import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# path
path = os.path.join(os.path.dirname(__file__), 'img3.png')

# captura
image = cv2.imread(path)


# alto, ancho, canal
alto, ancho, canal = image.shape

# preprocesamiento
if canal == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


plt.subplot(2, 2, 1)
ax = plt.imshow(image, cmap='gray')
ax.axes.set_axis_off()

# segmentacion
(umbral, binary_img) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.subplot(2, 2, 2)
ax = plt.imshow(binary_img, cmap='gray')
ax.axes.set_axis_off()

# Rellenar el objeto
(contornos, jerarquia) = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_img = np.zeros_like(binary_img)
cv2.drawContours(filled_img, contornos, -1, 255, cv2.FILLED)

plt.subplot(2, 2, 3)
ax = plt.imshow(filled_img, cmap='gray')
ax.axes.set_axis_off()

# Descripcion
contornos, _ = cv2.findContours(filled_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
(_, __, angulo) = cv2.minAreaRect(contornos[0])  # Ángulo en sentido contrario a las agujas del reloj


# Rotacion
angulo_rotada = angulo - 90
rotacion = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo_rotada, 1)
rotada = cv2.warpAffine(image, rotacion, (ancho, alto))

plt.subplot(2, 2, 4)
ax = plt.imshow(rotada, cmap='gray')
ax.axes.set_axis_off()


# mostrar imagen
plt.show()
