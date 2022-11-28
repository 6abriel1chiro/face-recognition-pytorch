import cv2
import numpy as np

if  __name__ == '__main__':
    img = cv2.imread('imagen.jpeg')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gen_faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    candidates = gen_faces.detectMultiScale(gray_img,scaleFactor = 1.1,minNeighbors=5,minSize=(60,60), flags = cv2.CASCADE_SCALE_IMAGE)

    for c in candidates:
        cv2.rectangle(img,(c[0], c[1]), (c[0] + c[2], c[1] + c[3]),(255,0,0), 2) 
    cv2.imshow("demo", img)
    cv2.waitKey()