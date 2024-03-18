import cv2 as cv
import numpy as np
import os
import time
import pickle
from mtcnn_ort import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FACELOADING:
    def __init__(self, directory, existing_Y):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.existing_Y = existing_Y  # Initialize existing_Y here
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        x, y, w, h = self.detector.detect_faces(img)[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, directory):
        FACES = []
        for im_name in os.listdir(directory):
            try:
                image_path = os.path.join(directory, im_name)  # Use os.path.join to ensure correct path construction
                single_face = self.extract_face(image_path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        existing_labels = set(self.existing_Y)
        print("Existing classes:", existing_labels)
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)  # Use os.path.join to ensure correct path construction
            if sub_dir not in existing_labels:  # Check if the class is new
                FACES = self.load_faces(path)
                labels = [sub_dir for _ in range(len(FACES))]
                print(len(labels))
                self.X.extend(FACES)
                self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)

data = np.load('faces_embeddings_done_4classes.npz')
existing_X, existing_Y = data['arr_0'], data['arr_1']
time1 = time.time()
faceloading = FACELOADING('./train', existing_Y)  # Pass existing_Y to FACELOADING
X, Y = faceloading.load_classes()
time2 = time.time()
print(f"{time2 - time1}s")

embedder = FaceNet()
detector = MTCNN()

# Function to get embedding of face image
def get_embedding(face_image):
    face_image = face_image.astype('float32')  # 3D (160x160x3)
    face_image = np.expand_dims(face_image, axis=0)
    yhat = embedder.embeddings(face_image)
    return yhat[0]  # 512D Image

# Compute embeddings for the new faces
EMBEDDED_X = []
for image in X:
    EMBEDDED_X.append(get_embedding(image))

EMBEDDED_X = np.asarray(EMBEDDED_X)

# Combine existing and new embeddings
combined_X = np.concatenate((existing_X, EMBEDDED_X))
combined_Y = np.concatenate((existing_Y, Y))

# Label Encoding of Images
encoder = LabelEncoder()
encoder.fit(combined_Y)
encoded_Y = encoder.transform(combined_Y)

# Splitting the data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(combined_X, encoded_Y, shuffle=True, random_state=20)

# Train the SVC model using flattened data
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# Predictions on training and testing data
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

# Evaluate model performance
train_accuracy = accuracy_score(Y_train, ypreds_train)
test_accuracy = accuracy_score(Y_test, ypreds_test)

print("Accuracy of the Training Model is:", train_accuracy)
print("Accuracy of the Testing Model is:", test_accuracy)

# Loading the pre-trained model using pickle
with open('face_recognition_model', 'wb') as f:
    pickle.dump((model, encoder), f)
