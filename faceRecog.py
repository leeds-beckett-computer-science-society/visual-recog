import cv2
import time


# utilizes haarcascade
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# enter in webcam path or 0,1,2,3
video = cv2.VideoCapture(0)

# use this variable for counting frames
a = 1

# loops through each video frame
while True:
    a = a+1

    # initialise video frames and print values
    check, frame = video.read()
    print(check)
    print(frame)

    # convert frame to grayscale (higher accuracy)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set parameter for how the algorithm approaches the frame
    # 1.05 means each iteration of the frame the image is decreased 5%
    faces = face_cascade.detectMultiScale(gray_img,
     scaleFactor=1.05,
     minNeighbors=5)

    # take position values from face_cascade and draw rectangle around face
    for x, y, w, h in faces:
        gray_img = cv2.rectangle(gray_img, (x, y), (x + w, y + h), (0, 255, 0), 3)


    # opens window to view webcam
    cv2.imshow("Capture", gray_img)

    # each iteration of loop waits 1 millisecond
    # can increase to have lesser framerate for optimization
    key = cv2.waitKey(1)

    # exit key
    if key == ord ('q'):
        break

print(a)
video.release(
cv2.destroyAllWindows()

