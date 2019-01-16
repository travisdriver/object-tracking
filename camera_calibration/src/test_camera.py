import numpy as np
import cv2
from imutils.video import VideoStream

#cap = cv2.VideoCapture(-1)
cap = VideoStream(src=0).start()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(True):
    frame = cap.read()
    #frame = cv2.flip(frame,0)

    # write the flipped frame
    #out.write(frame)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
