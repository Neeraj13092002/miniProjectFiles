import cv2 as cv
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import tkinter as tk
from tkinter import filedialog

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

def process_image(image_path):
    # Load the image
    test_image = cv.imread(image_path)
    test_image = cv.cvtColor(test_image, cv.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = detector.detect_faces(test_image)

    # Iterate over detected faces
    for face_info in faces:
        x, y, w, h = face_info['box']
        x2, y2 = x + w, y + h

        # Crop the face region
        face_region = test_image[y:y2, x:x2]

        # Resize the face region to 160x160
        face_region = cv.resize(face_region, (160, 160))

        # Get the FaceNet embedding for the face
        test_image_embed = get_embedding(face_region).reshape(1, -1)

        # Predict the class label using the loaded model
        class_label = encoder.inverse_transform(loaded_model.predict(test_image_embed))[0]

        # Draw a rectangle around the face
        cv.rectangle(test_image, (x, y), (x2, y2), (0, 255, 0), 2)

        # Write the class label on the image
        cv.putText(test_image, str(class_label), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with face detection and class labels
    cv.imshow('Test Image', cv.cvtColor(test_image, cv.COLOR_RGB2BGR))
    cv.waitKey(0)
    cv.destroyAllWindows()

def load_image_gui():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)

load_image_gui()
