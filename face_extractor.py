
import os
import cv2
import numpy as np

# Define paths
base_dir = os.path.dirname(__file__)
print(base_dir)
config_path = ( 'opencv_face_detector.pbtxt')
model_path = ( 'opencv_face_detector_uint8.pb')

# Read the model
model = cv2.dnn.readNetFromTensorflow(model_path, config_path)

# Create directory 'faces' if it does not exist
if not os.path.exists('faces'):
	print("New directory created")
	os.makedirs('faces')

# Loop through all images and strip out faces
count = 0
for file in os.listdir(base_dir + '/Images'):
	file_name, file_extension = os.path.splitext(file)
	if (file_extension in ['.png','.jpg']):
		image = cv2.imread( 'C:/Users/Adarsh/PycharmProjects/DriverDrowsy/Images/' + file)

		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

		model.setInput(blob)
		detections = model.forward()

		# Identify each face
		for i in range(0, detections.shape[2]):
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			confidence = detections[0, 0, i, 2]

			# If confidence > 0.5, save it as a separate file
			if (confidence > 0.5):
				count += 1
				frame = image[startY:endY, startX:endX]
				frame = cv2.resize(frame, (224,224))
				grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				equ_frame = cv2.equalizeHist(grayframe)
				cv2.imwrite('C:/Users/Adarsh/PycharmProjects/DriverDrowsy/faces/' + str(i) + '_' + file, equ_frame)

print("Extracted " + str(count) + " faces from all images")