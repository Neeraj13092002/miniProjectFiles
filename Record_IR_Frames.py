import cv2
from flask import Flask, Response
import pykinect_azure as pykinect
import numpy as np

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


def generate_frames():
    while True:
        # Get capture
        capture = device.update()

        # Get the infrared image
        ret, ir = capture.get_ir_image()
        temp = ir.astype(np.int32)
        ir_image = cv2.convertScaleAbs(temp, alpha=alpha, beta=beta)

        if not ret:
            continue

        # Encode the image as JPEG
        _, frame = cv2.imencode('.jpeg', ir_image)

        # Convert the frame to bytes
        frame_bytes = frame.tobytes()

        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def cleanup():
    device.stop_device()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=1050)
    except KeyboardInterrupt:
        cleanup()

