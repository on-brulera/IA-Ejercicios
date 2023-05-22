import cv2

# Leer imagen
imagen = cv2.imread('./percy.jpg')
# Mostrar imagen
cv2.imshow('Imagen', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

#leer un video

capture = cv2.VideoCapture('./goles.mp4')
while (capture.isOpened()):
    ret, frame = capture.read()
    if (ret == True):
        cv2.imshow("mundo1", frame)
        if (cv2.waitKey(30) == ord('s')):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()


#webcam

capture = cv2.VideoCapture(0)

while (capture.isOpened()):
    ret, frame = capture.read()
    cv2.imshow('webCam',frame)
    if (cv2.waitKey(1) == ord('s')):
        break

capture.release()
cv2.destroyAllWindows()