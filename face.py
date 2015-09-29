# A script that detects faces and not eyes

import cv2 as cv3
import numpy as np

# Main code
face_cascade = cv3.CascadeClassifier('/Users/ryan/opencv-3.0.0/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
# Get rid of the eye cascade
#eye_cascade = cv3.CascadeClassifier('/Users/ryan/opencv-3.0.0/data/haarcascades_cuda/haarcascade_eye.xml')
img_array = ['picture.jpg']
for imageIndex in img_array:
    img = cv3.imread(imageIndex)
    gray = cv3.cvtColor(img, cv3.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv3.rectangle(img,(x,y),(x+w,y+h),(89,235,171),2)
    cv3.imshow('img', img)
    cv3.waitKey(0)
    cv3.destroyAllWindows()