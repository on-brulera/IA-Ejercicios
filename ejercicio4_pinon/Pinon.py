import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import remove_small_holes, binary_erosion, disk
from skimage.measure import label

def pinon():
    # Leer la imagen
    g = imread('pinon.png')

    # Verificar si la imagen es RGB
    if g.shape[-1] == 3:
        # Convertir la imagen a escala de grises
        g = rgb2gray(g)

    # Mostrar la imagen original
    plt.subplot(2, 2, 1)
    plt.imshow(g, cmap='gray')
    plt.title('Imagen Original')

    # Binarizar la imagen
    bw = g > 128

    # Invertir la imagen binaria
    bw = ~bw

    # Rellenar los agujeros
    bw = remove_small_holes(bw, area_threshold=200)

    # Mostrar la imagen binaria
    plt.subplot(2, 2, 2)
    plt.imshow(bw, cmap='gray')
    plt.title('Imagen Binaria')

    # Aplicar la erosión morfológica
    selem = disk(10)  # Ajustar el tamaño del elemento estructurante según tus necesidades

    # Asegurarse de que bw y selem tengan la misma dimensionalidad
    if bw.ndim != selem.ndim:
        selem = selem[:, :, None]

    selem = selem.astype(bool)
    se = binary_erosion(bw, selem)

    # Mostrar la imagen erosionada
    plt.subplot(2, 2, 3)
    plt.imshow(se, cmap='gray')
    plt.title('Imagen Erosionada')

    # Restar se a bw
    J = bw.astype(int) - se.astype(int)
    J = J > 0

    # Convertir J en UMat
    J_umat = J.astype(np.uint8)

    # Mostrar la imagen final
    plt.subplot(2, 2, 4)
    plt.imshow(J.astype(float), cmap='gray')  # Convertir J a float antes de mostrarlo
    plt.title('Imagen Final')

    # Etiquetar los componentes conectados
    _, labels = cv2.connectedComponents(J_umat)

    # Calcular el número de dientes
    num_dientes = len(np.unique(labels)) - 1

    # Imprimir el número de dientes
    print('Número de dientes del piñón:', num_dientes)

if __name__ == '__main__':
    pinon()
    plt.tight_layout()
    plt.show()
