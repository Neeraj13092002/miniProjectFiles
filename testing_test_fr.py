import cv2 as cv
import numpy as np
import pickle
import mediapipe as mp
import time
from keras_facenet import FaceNet

alpha = 0.2  # Contrast control
beta = 0.009  # Brightness control

# Load the pre-trained model using pickle
with open('face_recognition_model', 'rb') as f:
    loaded_model, encoder = pickle.load(f)

# Load the FaceNet embedder
embedder = FaceNet()

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


# Function to get face embedding
def get_embedding(face_image):
    face_image = face_image.astype('float32')  # 3D (160x160x3)
    face_image = np.expand_dims(face_image, axis=0)
    yhat = embedder.embeddings(face_image)
    return yhat[0]  # 512D Image


# Function to process each frame
def process_frame(frame):
    labels = []
    # Convert the frame to RGB
    frame_rgb = frame

    # Run MediaPipe face detection
    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_region = frame[y:y + h, x:x + w]

            try:
                face_region = cv.resize(face_region, (160, 160))
                test_image_embed = get_embedding(face_region).reshape(1, -1)
                class_label = encoder.inverse_transform(loaded_model.predict(test_image_embed))[0]
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, str(class_label), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                labels.append(class_label)
                cv.imshow("Image",frame)
                cv.waitKey(1)
            except cv.error as e:
                print("Error:", e)
                continue

    else:
        print("No faces detected")
    return labels


# Main function for real-time face recognition
def real_time_face_recognition(frame):
    ir_image = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    temp = ir_image.astype(np.int32)
    frame = cv.convertScaleAbs(temp, alpha=alpha, beta=beta)
    class_labels = process_frame(frame)
    return class_labels
