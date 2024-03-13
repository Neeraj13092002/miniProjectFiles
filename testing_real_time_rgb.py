import cv2 as cv
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from datetime import datetime
# Load the pre-trained model using pickle
with open('face_recognition_model', 'rb') as f:
    loaded_model, encoder = pickle.load(f)

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
    print(faces)
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
    cap = cv.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break

        process_frame(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()  # Release the webcam
    cv.destroyAllWindows()

real_time_face_recognition()
