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


def select_face_with_least_x_value(faces):
    # Ensure there are faces detected
    if len(faces) == 0:
        return None

    # Sort the faces based on x value
    sorted_faces = sorted(faces, key=lambda x: x['box'][0])

    # Select the face with the least x value
    selected_face = sorted_faces[0]

    return selected_face


def save_selected_face(image, selected_face, index):
    if selected_face is None:
        return

    # Extract face coordinates
    x, y, w, h = selected_face['box']

    # Draw bounding box on the image
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Generate filenames based on index
    image_filename = f'selected_face_{index}.jpg'
    txt_filename = f'selected_face_{index}.txt'

    # Write the coordinates to a text file
    with open(txt_filename, 'w') as txt_file:
        txt_file.write(f'x: {x}, y: {y}, w: {w}, h: {h}')

    # Save the image with bounding box
    cv.imwrite(image_filename, image)


def process_frame(frame, index):
    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Select the face with the least x value
    selected_face = select_face_with_least_x_value(faces)

    # Save the selected face to a new file
    save_selected_face(frame.copy(), selected_face, index)

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
    index = 0
    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break

        process_frame(frame, index)
        index += 1

        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()  # Release the webcam
    cv.destroyAllWindows()


real_time_face_recognition()
