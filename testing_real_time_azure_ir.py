import os
import pykinect_azure as pykinect
import cv2 as cv
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet
import threading
# Load the pre-trained model using pickle
with open('face_recognition_model', 'rb') as f:
    loaded_model, encoder = pickle.load(f)

alpha = 0.2  # Contrast control
beta = 0.009  # Brightness control

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

# Start device
device = pykinect.start_device(config=device_config)

# Load the FaceNet embedder
embedder = FaceNet()

# Load the MTCNN detector
detector = MTCNN()

def get_embedding(face_image):
    face_image = face_image.astype('float32')  # 3D (160x160x3)
    face_image = np.expand_dims(face_image, axis=0)
    yhat = embedder.embeddings(face_image)
    return yhat[0]  # 512D Image

def process_frame(frame):
    # Detect faces in the frame

    faces = detector.detect_faces(frame)
    # Iterate over detected faces
    for face_info in faces:
        x, y, w, h = face_info['box']
        x2, y2 = x + w, y + h
        # Crop the face region
        face_region = frame[y:y2, x:x2]
        # Resize the face region to 160x160
        face_region = cv.resize(face_region, (160, 160))
        # Get the FaceNet embedding for the face
        test_image_embed = get_embedding(face_region).reshape(1, -1)
        # Predict the class label using the loaded model
        class_label = encoder.inverse_transform(loaded_model.predict(test_image_embed))[0]
        # Draw a rectangle around the face
        cv.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        # Write the class label on the frame
        cv.putText(frame, str(class_label), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Display the frame with face detection and class labels
    cv.imshow('Real-Time Face Recognition', frame)



def real_time_face_recognition():
    # Open webcam
    while True:
        capture = device.update()

        # Get the infrared image
        ret, ir = capture.get_ir_image()
        # print(ir.shape)
        ir_image = cv.cvtColor(ir, cv.COLOR_GRAY2BGR)
        temp = ir_image.astype(np.int32)
        frame = cv.convertScaleAbs(temp, alpha=alpha, beta=beta)
        # print(frame.shape)
        # cv.imshow("img",frame)
      #  ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            pass
        process_frame(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break
    cv.destroyAllWindows()

readimagethread = threading.Thread(target=real_time_face_recognition)
readimagethread.start()

#real_time_face_recognition()
