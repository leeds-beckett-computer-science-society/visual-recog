import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


# take data from visualrec and format it into graph
def animate(i):
    graph_data = open('data.txt', 'r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    ax1.clear()
    ax1.plot(xs, ys)


def visualrec():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video = cv2.VideoCapture(0)
    i = 0

    while True:
        i = i + 1
        check, frame = video.read()
        frame = cv2.resize(frame, (320, 240))
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img,
                                              scaleFactor=1.05,
                                              minNeighbors=5)

        if len(faces) == 0:
            print("No faces found :(")
        else:
            print("Number of faces detected: " + str(faces.shape[0]))

            # write current number of faces on screen to file for graph plotting
            f = open('data.txt', "a")
            f.write(str(i) + "," + str(faces.shape[0]) + '\n')

        # take position values from face_cascade and draw rectangle around face
        for x, y, w, h in faces:
            gray_img = cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.putText(gray_img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(gray_img, "Number of faces detected: " + str(faces.shape[0]), (0, gray_img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # opens window to view webcam
        cv2.imshow("idiot on screen", gray_img)
        key = cv2.waitKey(1)

        # exit key
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


ani = animation.FuncAnimation(fig, animate, interval=1000)

visualrec()
plt.show()
