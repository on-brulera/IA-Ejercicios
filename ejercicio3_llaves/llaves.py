import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
imagen = cv2.imread('./llaves.png')

# Convertir a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Binarizar la imagen
_, imagen_bin = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Aplicar operaciones de morfología matemática
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
imagen_morf = cv2.morphologyEx(imagen_bin, cv2.MORPH_OPEN, se)

# Etiquetar los componentes conectados
_, labels = cv2.connectedComponents(imagen_morf)
num_llaves = np.max(labels)

# Mostrar la imagen binarizada y el resultado de la morfología matemática
plt.subplot(1, 2, 1)
plt.imshow(imagen_bin, cmap='gray')
plt.title('Imagen binarizada')

plt.subplot(1, 2, 2)
plt.imshow(imagen_morf, cmap='gray')
plt.title(f'Número de llaves inglesas: {num_llaves}')

plt.show()