import cv2 as cv3
import sys

cascPath = sys.argv[1]
faceCascade = cv3.CascadeClassifier('/Users/ryan/opencv-3.0.0/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
eyeCascade = cv3.CascadeClassifier('/Users/ryan/opencv-3.0.0/data/haarcascades/haarcascade_eye.xml')
video_capture = cv3.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv3.cvtColor(frame, cv3.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv3.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv3.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv3.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    # Display the resulting frame
    cv3.imshow('Video', frame)

    if cv3.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv3.destroyAllWindows()