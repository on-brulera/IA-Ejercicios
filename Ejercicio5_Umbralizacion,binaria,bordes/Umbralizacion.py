import cv2

def umbralizacion(imagen, umbral):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar umbralización
    _, imagen_umbralizada = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)

    return imagen_umbralizada

# Cargar la imagen
imagen_original = cv2.imread('C:/Users/tvo/Downloads/rosa.jpg')

# Mostrar la imagen original
cv2.imshow('Imagen Original', imagen_original)
cv2.waitKey(0)

# Especificar el umbral (0-255)
umbral = 128

# Aplicar umbralización
imagen_umbralizada = umbralizacion(imagen_original, umbral)

# Convertir la imagen umbralizada a formato BGR
imagen_umbralizada_bgr = cv2.cvtColor(imagen_umbralizada, cv2.COLOR_GRAY2BGR)

# Mostrar la imagen umbralizada
cv2.imshow('Imagen Umbralizada', imagen_umbralizada_bgr)
cv2.waitKey(0)

# Cerrar las ventanas de imagen
cv2.destroyAllWindows()
