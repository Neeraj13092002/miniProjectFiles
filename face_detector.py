import cv2 as cv
import os
from mtcnn.mtcnn import MTCNN

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
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
                image_path = os.path.join(directory, im_name)
                single_face = self.extract_face(image_path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        X = []
        Y = []
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            X.extend(FACES)
            Y.extend(labels)
        return X, Y

    def plot_images(self,X,Y):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 12))
        for num, image in enumerate(X):
            ncols = 3
            nrows = (len(Y) // ncols) + 1
            plt.subplot(nrows, ncols, num+1)
            plt.imshow(image)
            plt.axis('off')
        plt.show()

# Example usage:
#Its training on the Images in the train folder
faceloading = FACELOADING('./train')
X, Y = faceloading.load_classes()
faceloading.plot_images(X,Y)
