import cv2
import os
import numpy as np

if not os.path.exists('Images'):
	print("New directory created")
	os.makedirs('Images')

cap = cv2.VideoCapture(0)
n=0
while True:
    ret, frame = cap.read()
    if (n < 200):
        cv2.imshow('cam', frame)
        frame = cv2.resize(frame, (224, 224))
        file_name_path = './Images/' + str(n) + '.jpg'
        cv2.imwrite(file_name_path, frame)
        n = n+1
    if cv2.waitKey(1) == 13 or n == 140:
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")