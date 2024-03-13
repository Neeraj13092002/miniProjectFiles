import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FACELOADING:
    def __init__(self,directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename, output_dir='output_2'):
        os.makedirs(output_dir, exist_ok=True)
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width, channels = img.shape
        faces = self.detector.detect_faces(img)
        if faces:
            # Sort faces based on x-coordinate
            faces.sort(key=lambda x: x['box'][0])

            # Extract the face with the lowest x-value
            x, y, w, h = faces[0]['box']
            # print(type(x))
            x, y = abs(x), abs(y)
            print("Coming")
            face = img[y:y + h, x:x + w]
            face_arr = cv.resize(face, self.target_size)
            # Draw bounding box around the face
            # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            answer = self.yolo_normalize(x,y,w,h,width,height)
            # Save the new image with bounding box
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            txt_filename = os.path.join(output_dir, base_filename + '.txt')
            # Save x, y, w, h coordinates to text file
            print(answer)
            with open(txt_filename, 'w') as f:
                f.write(f'x: {answer[0]}, y: {answer[1]}, w: {answer[2]}, h: {answer[3]}')

            return face_arr
        else:
            return None

    def load_faces(self, directory):
        FACES = []
        for im_name in os.listdir(directory):
            try:
                image_path = directory + im_name
                single_face = self.extract_face(image_path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory + '/' + sub_dir + '/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(len(labels))
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)

    def plot_images(self):
        plt.figure(figsize = (16,12))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = (len(self.Y)//ncols) + 1
            plt.subplot(nrows, ncols, num+1)
            plt.imshow(image)
            plt.axis('off')
        plt.show()

    def yolo_normalize(self, x, y, w, h, img_width, img_height):
        # Convert inputs to a numpy array
        data = np.array([x, y, w, h])

        # Normalize x and w relative to image width
        x_normalized = x / img_width
        w_normalized = w / img_width

        # Normalize y and h relative to image height
        y_normalized = y / img_height
        h_normalized = h / img_height

        return [x_normalized, y_normalized, w_normalized, h_normalized]


faceloading = FACELOADING('./valid')
X,Y = faceloading.load_classes()
faceloading.plot_images()

embedder = FaceNet()
detector = MTCNN()
