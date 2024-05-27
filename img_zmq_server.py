import cv2
import zmq
import pykinect_azure as pykinect

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

# Start device
device = pykinect.start_device(config=device_config)

# Initialize ZeroMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")  # Bind to port for broadcasting

alpha = 0.2  # Contrast control
beta = 0.009  # Brightness control


def generate_frames(frame_type):
    while True:
        # Get capture
        capture = device.update()

        ret_ir, ir = capture.get_ir_image()  # IR image
        ret_rgb, rgb = capture.get_color_image()  # RGB image

        # call convertScaleAbs function, just for visualisation
        adjusted = cv2.convertScaleAbs(ir, alpha=alpha, beta=beta)

        if not ret_ir or not ret_rgb:
            continue

        # Convert IR image to grayscale
        # ir_gray = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)

        # Compress the frame using JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Adjust quality as needed
        _, ir_jpeg = cv2.imencode('.jpg', adjusted, encode_param)
        _, rgb_jpeg = cv2.imencode('.jpg', rgb, encode_param)

        # Send the compressed frames
        socket.send_multipart([b"IR", ir_jpeg])
        socket.send_multipart([b"RGB", rgb_jpeg])


# Start generating frames
generate_frames("RGB")
