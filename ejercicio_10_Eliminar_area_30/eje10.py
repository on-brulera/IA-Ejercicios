import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen y convertirla a escala de grises
I = cv2.imread('figurasn.png')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# Binarizar la imagen
_, I = cv2.threshold(I, 128, 255, cv2.THRESH_BINARY)

# Mostrar la imagen original
plt.subplot(2, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Imagen original')

# Cambiar el color
I = cv2.bitwise_not(I)

# Mostrar la imagen con el color cambiado
plt.subplot(2, 2, 2)
plt.imshow(I, cmap='gray')
plt.title('Color cambiado')

# Etiquetar los objetos en la imagen binaria
_, etiquetas = cv2.connectedComponents(I)

# Obtener las propiedades de los objetos etiquetados
propiedades = cv2.connectedComponentsWithStats(I)

# Obtener las áreas de los objetos etiquetados
areas = propiedades[2][:, cv2.CC_STAT_AREA]

# Ordenar las áreas de manera descendente
indices = np.argsort(areas)[::-1]

# Encontrar el área del objeto más grande
area_objeto_mas_grande = areas[indices[0]]

# Calcular el umbral para eliminar objetos pequeños
umbral = area_objeto_mas_grande * (3/100)

# Eliminar los objetos cuya área sea menor al umbral
for i in range(1, len(areas)):
    if areas[i] < umbral:
        etiquetas[etiquetas == i] = 0

# Mostrar la imagen resultante sin los objetos eliminados
imagen_resultante = etiquetas > 0
plt.subplot(2, 2, 3)
plt.imshow(imagen_resultante, cmap='gray')
plt.title('Imagen resultante')

# Mostrar las subtramas
plt.tight_layout()
plt.show()
