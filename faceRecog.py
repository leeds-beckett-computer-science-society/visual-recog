import cv2


#

def visualrec():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video = cv2.VideoCapture(0)

    while True:
        check, frame = video.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img,
                                              scaleFactor=1.05,
                                              minNeighbors=5)

        # take position values from face_cascade and draw rectangle around face
        for x, y, w, h in faces:
            gray_img = cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.putText(gray_img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

        # opens window to view webcam
        cv2.imshow("idiot on screen", gray_img)
        key = cv2.waitKey(1)

        # exit key
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


visualrec()
