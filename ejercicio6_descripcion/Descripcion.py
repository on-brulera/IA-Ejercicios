import cv2
import numpy as np

# Leer la imagen
image = cv2.imread("imades.jpg", cv2.IMREAD_GRAYSCALE)

# Binarizar la imagen
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Encontrar los contornos de los objetos
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Etiquetar los objetos y dibujar el BoundingBox y Centroid
for i, contour in enumerate(contours):
    # Calcular el BoundingBox
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calcular el Centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:  # Verificar que el área no sea cero
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

# Mostrar la imagen con los objetos etiquetados
cv2.imshow("Etiquetado de objetos", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
