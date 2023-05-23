import cv2
import matplotlib.pyplot as plt
import numpy as np

g = cv2.imread('gas2.png')
if len(g.shape) == 3:  # Es RGB?
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

plt.subplot(2, 2, 1)
plt.imshow(g, cmap='gray')

# Saca el borde de una imagen en escala de grises
bw = cv2.Canny(g, 50, 150)
plt.subplot(2, 2, 2)
plt.imshow(bw, cmap='gray')

bw = g > 128  # Binarizamos
bw2 = cv2.fillHoles(np.uint8(bw))
bw2 = ~bw2  # Cambiar color
plt.subplot(2, 2, 3)
plt.imshow(bw2, cmap='gray')

plt.show()
