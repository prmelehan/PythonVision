# A script that can detect a face and then use my PythonDB scripts to match it to a name and make is searchable
import cv2 as cv3
import numpy as np


face_cascade = cv3.CascadeClassifier('/Users/ryan/opencv-3.0.0/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
eye_cascade = cv3.CascadeClassifier('/Users/ryan/opencv-3.0.0/data/haarcascades_cuda/haarcascade_eye.xml')
img_array = img_array = ['picture.jpg']
for imageIndex in img_array:
    img = cv3.imread(imageIndex)
    gray = cv3.cvtColor(img, cv3.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv3.rectangle(img,(x,y),(x+w,y+h),(89,235,171),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv3.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    cv3.imshow('img', img)
    cv3.waitKey(0)
    cv3.destroyAllWindows()
    


