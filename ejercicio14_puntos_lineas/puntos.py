import cv2
import os
import matplotlib.pyplot as plt

# path
path = os.path.join(os.path.dirname(__file__), 'puntos_lineas.jpg')

# captura
image = cv2.imread(path)

# preprocesamiento
_, __, canal = image.shape

if canal == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Transformar la imagen a binaria
(umbral, binary_img) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Aplicar la operación de apertura
shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
result = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, shape)

# Encontrar los contornos de los puntos
contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Contar el número de puntos
num_puntos = len(contours)

# Mostrar la imagen y el número de puntos
plt.subplot(1, 2, 1)
plt.title('Imagen binaria')
ax = plt.imshow(binary_img, cmap='gray')
ax.axes.set_axis_off()

plt.subplot(1, 2, 2)
plt.title(f'Número de puntos: {num_puntos}')
ax = plt.imshow(result, cmap='gray')
ax.axes.set_axis_off()

# Mostrar la imagen
plt.show()
