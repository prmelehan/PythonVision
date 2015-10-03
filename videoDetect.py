import cv2 as cv3
import sys

vidPath = sys.argv[1]
faceCascade = cv3.CascadeClassifier('/Users/ryan/opencv-3.0.0/data/lbpcascades/lbpcascade_frontalface.xml')

video_capture = cv3.VideoCapture(vidPath)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv3.cvtColor(frame, cv3.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv3.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv3.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv3.imshow('Video', frame)

    if cv3.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv3.destroyAllWindows()