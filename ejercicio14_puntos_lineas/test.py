import cv2
import os
import numpy as np

# Cargar la imagen binaria
image = cv2.imread(os.path.join(os.path.dirname(__file__), 'lineas.jpg'), 0)

# Aplicar la operación de apertura para eliminar ruido y fragmentos pequeños
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Detectar bordes en la imagen
edges = cv2.Canny(opening, 50, 150)

# Aplicar la transformada de Hough probabilística para detectar líneas
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Filtrar líneas diagonales
filtered_lines = []
angle_threshold = 30

for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    if abs(angle - 50) < angle_threshold or abs(angle + 50) < angle_threshold:
        filtered_lines.append(line)

# Crear una imagen en blanco del mismo tamaño que la imagen original
result = np.zeros_like(image)

# Dibujar las líneas diagonales en la imagen de resultado
for line in filtered_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(result, (x1, y1), (x2, y2), 255, 2)

# Contar las líneas diagonales
num_diagonal_lines = len(filtered_lines)

# Mostrar los resultados
cv2.imshow('Imagen original', image)
cv2.imshow('Bordes', edges)
cv2.imshow('Imagen de resultado', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Número de líneas diagonales:', num_diagonal_lines)
