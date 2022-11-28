import cv2
import torchvision.transforms as transforms
import numpy as np

model_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
dim = (64, 64)

i=0
while True:
    ret, img = video.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    candidates = model_haar.detectMultiScale(gray_img,scaleFactor = 1.1,minNeighbors=5,minSize=(60,60), flags = cv2.CASCADE_SCALE_IMAGE)


    for c in candidates:
        cv2.rectangle(img,(c[0], c[1]), (c[0] + c[2], c[1] + c[3]),(255,0,0), 2) 
        
        resized = cv2.resize(gray_img[  c[1]: c[1]+c[3] , c[0] : c[0]+c[2]  ], dim, interpolation = cv2.INTER_AREA)
        start_row = 0
        end_row  = 64
        start_col = 6
        end_col = 58
        cropped = resized[start_row:end_row, start_col:end_col]
        #cv2.imshow("cropped", cropped)
        resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)

        #print('Resized Dimensions : ',resized.shape)

        cv2.imshow("resized", resized)
        cv2.imwrite('img/gabriel'+str(i)+'.jpg',resized )
        i+=1
        print(i)
        if i > 300: break
    cv2.imshow("demo", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


