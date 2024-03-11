import cv2
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import pykinect_azure as pykinect

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
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Iterate over detected faces
    for face_info in faces:
        x, y, w, h = face_info['box']
        x2, y2 = x + w, y + h

        # Crop the face region
        face_region = gray[y:y2, x:x2]

        # Resize the face region to 160x160
        face_region = cv2.resize(face_region, (160, 160))

        # Get the FaceNet embedding for the face
        test_image_embed = get_embedding(face_region).reshape(1, -1)

        # Predict the class label using the loaded model
        class_label = encoder.inverse_transform(loaded_model.predict(test_image_embed))[0]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        # Write the class label on the frame
        cv2.putText(frame, str(class_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

def process_ir_frames():
    # Initialize the Kinect camera
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    # Start the device
    device = pykinect.start_device(config=device_config)

    while True:
        # Get the capture
        capture = device.update()

        # Get the IR frame
        ret, ir_frame = capture.get_ir_image()
        if not ret:
            continue

        # Convert the IR frame to BGR format
        ir_frame_bgr = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)

        # Process the frame
        processed_frame = process_frame(ir_frame_bgr)

        # Display the frame with facial recognition
        cv2.imshow(processed_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the device
    device.stop_device()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_ir_frames()
