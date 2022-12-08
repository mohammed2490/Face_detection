import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# img = cv.imread("images/lady.jpg")
# cv.imshow("Lady", img) 

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read() 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv.imshow("Video", frame)

    if cv.waitKey(20) & 0xFF==ord("d"):
        break

capture.release()
cv.destroyAllWindows()

# haar_cascade = cv.CascadeClassifier('haar_face.xml')

# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
# print(faces_rect)

# for (x,y,w,h) in faces_rect:
#     cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# cv.imshow('detected faces', img)
# print(img.shape)


# cv. waitKey(0)