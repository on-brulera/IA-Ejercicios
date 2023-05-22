import cv2
import matplotlib.pyplot as plt
import numpy as np

#VISUALIZAR

def mostrarImagenes(texto, imagen, texto2, imagen2):
    plt.subplot(2, 1, 1)
    plt.title(texto)
    plt.imshow(imagen)
    plt.subplot(2, 1, 2)
    plt.title(texto2)
    plt.imshow(imagen2)
    plt.tight_layout()
    plt.show()

#---AUMENTAR CONTRASTE---

# Leer imagen normal
imagen = cv2.imread('./rueda.jpg')

#Imagen en escala de Gris

imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aumentar el contraste
alpha = 1.5  # Factor de contraste
beta = 50    # Factor de brillo
imagenContraste = np.clip(alpha * imagenGris + beta, 0, 255).astype(np.uint8)


# Mostrar imagen original y con contraste
# mostrarImagenes("Imagen Original", imagen, "Imagen Contraste", imagenContraste);




#---CONTRASTE USANDO LA ECUACIÓN DEL HISTOGRAMA---

#leer en escala de grises usando metodo de OpenCV
imagenEG = cv2.imread("./rueda.jpg",0)

# Aplicar la ecualización de histograma
imagenEqualizada = cv2.equalizeHist(imagenEG)

# Mostrar imagen original y con contraste
# mostrarImagenes("Imagen Original", imagen, "Imagen Ecualizada", imagenEqualizada);





#---CONVERTIR RGB A ESCALA DE GRISES
imagenRGB =  cv2.imread('./ruedaRGB.jpg');

# Convertir la imagen RGB a escala de grises
imagenGris = cv2.cvtColor(imagenRGB, cv2.COLOR_BGR2GRAY)

# mostrarImagenes("Imagen Original RGB", imagenRGB, "Imagen en Escala Gris", imagenGris);


#CONVERTIR RGB A EQUIVALENTE HSV

imagenHSV = cv2.cvtColor(imagenRGB, cv2.COLOR_BGR2HSV)
mostrarImagenes("Imagen Original RGB", imagenRGB, "Imagen HSV", imagenHSV);