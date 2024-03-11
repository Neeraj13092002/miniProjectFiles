import cv2
from flask import Flask, Response
import pykinect_azure as pykinect
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet
import threading
import time

app = Flask(__name__)

alpha = 0.2  # Contrast control
beta = 0.009  # Brightness control

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

# Start device
device = pykinect.start_device(config=device_config)

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


def process_frame(ir_frame):
    # Convert IR frame to BGR for visualization
    ir_frame_bgr = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)

    # Detect faces in the frame
    timestart = time.time()
    faces = detector.detect_faces(ir_frame_bgr)
    timeend = time.time()
    print("time taken to execute", timeend - timestart)

    # Iterate over detected faces
    for face_info in faces:
        x, y, w, h = face_info['box']
        x2, y2 = x + w, y + h
        # Crop the face region
        face_region = ir_frame[y:y2, x:x2]
        # Resize the face region to 160x160
        face_region = cv2.resize(face_region, (160, 160))
        # Get the FaceNet embedding for the face
        test_image_embed = get_embedding(face_region).reshape(1, -1)
        # Predict the class label using the loaded model
        class_label = encoder.inverse_transform(loaded_model.predict(test_image_embed))[0]
        # Draw a rectangle around the face
        cv2.rectangle(ir_frame_bgr, (x, y), (x2, y2), (0, 255, 0), 2)
        # Write the class label on the frame
        cv2.putText(ir_frame_bgr, str(class_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with face detection and class labels
    cv2.imshow('Real-Time Face Recognition', ir_frame_bgr)
    cv2.waitKey(1)


def real_time_face_recognition():
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break
        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()


def generate_frames():
    while True:
        # Get capture
        capture = device.update()

        # Get the infrared image
        ret, ir = capture.get_ir_image()
        ir_image = ir.astype(np.int32)

        if not ret:
            continue

        # Apply contrast and brightness adjustment
        ir_image = cv2.convertScaleAbs(ir_image, alpha=alpha, beta=beta)

        # Yield the frame in the response
        _, frame = cv2.imencode('.jpeg', ir_image)
        frame_bytes = frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    read_image_thread = threading.Thread(target=real_time_face_recognition)
    read_image_thread.start()
    app.run(host='0.0.0.0', port=1050)
