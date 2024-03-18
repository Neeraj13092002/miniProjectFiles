import cv2 as cv
import numpy as np
import os
from mtcnn_ort import MTCNN
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set ORT_DEVICE to specify the target device (CPU or GPU)
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    # GPU is available, use GPU
    os.environ['ORT_DEVICE'] = 'cuda'
else:
    # No GPU available, use CPU
    os.environ['ORT_DEVICE'] = 'cpu'
class FACELOADING:
    def __init__(self,directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        height, width, channels = img.shape

        start_time = time.time()
        faces = self.detector.detect_faces(img)
        end_time = time.time()
        print(f"{end_time - start_time:.2f} seconds")

        if faces:
            # Sort faces based on x-coordinate
            faces.sort(key=lambda x: x['box'][0])
            # Extract the face with the lowest x-value
            x, y, w, h = faces[0]['box']
            x, y = abs(x), abs(y)
            answer = self.yolo_normalize(x, y, w, h, width, height)
            base_filename = os.path.splitext(os.path.basename(filename))[0]

            #Use this part if you want to check the box position accuracy
            # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # img_filename = os.path.join(output_dir, base_filename + '.jpg')
            # cv.imwrite(img_filename, cv.cvtColor(img, cv.COLOR_RGB2BGR))

            txt_filename = os.path.join(output_dir, base_filename + '.txt')
            # Save x, y, w, h coordinates to text file
            print(answer)
            with open(txt_filename, 'w') as f:
                f.write(f'x: {answer[0]}, y: {answer[1]}, w: {answer[2]}, h: {answer[3]}')

    def load_images(self):
        image_dir = os.path.join(os.getcwd(), self.directory)  # Get the full path of the directory
        for im_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, im_name)
            self.extract_face(image_path)

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

time3 = time.time()
faceloading = FACELOADING('./valid/Raafay')
faceloading.load_images()
time4 = time.time()
print(f"Total time taken for All images is:{time4-time3}")
