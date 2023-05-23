import cv2
from pyzbar import pyzbar

# Cargar la imagen
image = cv2.imread("barra2.jpg")

# Buscar los códigos de barras en la imagen
barcodes = pyzbar.decode(image)

# Imprimir los números ASCII de los códigos de barras
for barcode in barcodes:
    barcode_data = barcode.data.decode("utf-8")  # Decodificar los datos del código de barras a una cadena Unicode
    ascii_numbers = [ord(char) for char in barcode_data]  # Obtener los números ASCII de cada carácter en el código de barras
    print(ascii_numbers)
