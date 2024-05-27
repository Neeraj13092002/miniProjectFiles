import cv2 as cv
import numpy as np
import pickle
from mtcnn_ort import MTCNN
import time
from keras_facenet import FaceNet
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://172.1.40.74:5555")  # Connect to the sender's address

# Subscribe to all messagess
socket.setsockopt_string(zmq.SUBSCRIBE, "")

# Set to 1 if rotation by 180 degree is needed.
rotate_flag = 0

# Load the pre-trained model using pickle
with open('face_recognition_model', 'rb') as f:
    loaded_model, encoder = pickle.load(f)

# Load the FaceNet embedder
embedder = FaceNet()

# Load the MTCNN detector
detector = MTCNN()
cap = cv.VideoCapture(0)  # Open webcam


# Function to get face embedding
def get_embedding(face_image):
    face_image = face_image.astype('float32')  # 3D (160x160x3)
    face_image = np.expand_dims(face_image, axis=0)
    yhat = embedder.embeddings(face_image)
    return yhat[0]  # 512D Image


# Function to process each frame
def process_frame(frame):
    labels = []
    # Detect faces in the frame
    faces = detector.detect_faces(frame)
    for face_info in faces:
        x, y, w, h = face_info['box']
        x2, y2 = x + w, y + h
        face_region = frame[y:y2, x:x2]
        face_region = cv.resize(face_region, (160, 160))
        test_image_embed = get_embedding(face_region).reshape(1, -1)
        class_label = encoder.inverse_transform(loaded_model.predict(test_image_embed))[0]
        labels.append(class_label)
        cv.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, str(class_label), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv.imshow('Real-Time Face Recognition', frame)
    return labels


# Main function for real-time face recognition
def real_time_face_recognition():
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        pass

    if rotate_flag:
        # Rotate the frame by 180 degree
        frame = cv.rotate(frame, cv.ROTATE_180)

    class_labels = process_frame(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        pass

    return class_labels


# Call the real-time face recognition function
class_labels = real_time_face_recognition()
