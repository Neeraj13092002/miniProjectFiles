import cv2 as cv
import numpy as np
import pickle
import mediapipe as mp
import time
from keras_facenet import FaceNet
from pathlib import Path
import pykinect_azure as  pykinect

rotate_flag = 1

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
# device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_YUY2
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
# print(device_config)
# Start device
device = pykinect.start_device(config=device_config)

alpha = 0.2  # Contrast control
beta = 0.009  # Brightness control

# Load the pre-trained model using pickle
with open('face_recognition_model', 'rb') as f:
    loaded_model, encoder = pickle.load(f)

# Load the FaceNet embedder
embedder = FaceNet()

# Load the Mediapipe detector
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def get_embedding(face_image):
    face_image = face_image.astype('float32')  # 3D (160x160x3)
    face_image = np.expand_dims(face_image, axis=0)
    yhat = embedder.embeddings(face_image)
    return yhat[0]  # 512D Image


def process_frame(frame):
    # Convert the frame to RGB (MediaPipe requires RGB input)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detect faces in the frame
    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            x, y, w, h = bbox
            x2, y2 = x + w, y + h
            face_region = frame[y:y2, x:x2]
            face_region = cv.resize(face_region, (160, 160))
            test_image_embed = get_embedding(face_region).reshape(1, -1)
            class_label = encoder.inverse_transform(loaded_model.predict(test_image_embed))[0]
            print(class_label)
            cv.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, str(class_label), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv.imshow('Real-Time Face Recognition', frame)

def real_time_face_recognition():
    while True:
        capture = device.update()
        ret_ir, ir  = capture.get_ir_image()

        if not ret_ir:
            continue

        adjusted = ir
        if rotate_flag:
            # Rotate the frame by 180 degree
            adjusted = cv.rotate(adjusted, cv.ROTATE_180)

        process_frame(adjusted)

real_time_face_recognition()
