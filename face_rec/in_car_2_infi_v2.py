import torch
import cv2
import os
import time
import socket as sc
import test_fr
import pykinect_azure as pykinect

# Set to 1 if rotation by 180 degree is needed.
rotate_flag = 0

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

# Create a UDP socket
sock = sc.socket(sc.AF_INET, sc.SOCK_DGRAM)

# Set the broadcast address
broadcast_address = '172.1.40.24'
port = 3732

# Bind the socket to the port
sock.bind(('0.0.0.0', port))

# Enable broadcasting mode
sock.setsockopt(sc.SOL_SOCKET, sc.SO_BROADCAST, 1)

seat_belt_detection_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                           path=r'D:\jetson_models_test\best.pt',
                                           force_reload=True)


def fc_rec():
    while True:
        # Call the real-time face recognition function from the imported script
        class_labels = test_fr.real_time_face_recognition()

        if 'Raafay' in class_labels:
            print(f'Welcome, {class_labels}!')
            print()
            break


def sb(time_to_capture):
    driver_with_sb_count = 0
    copassenger_with_sb_count = 0
    start_time = time.time()

    while (time.time() - start_time) < time_to_capture:
        # Get capture
        capture = device.update()
        ret_ir, ir = capture.get_ir_image()  # IR image

        # call convertScaleAbs function, just for visualisation
        adjusted = cv2.convertScaleAbs(ir, alpha=alpha, beta=beta)

        if not ret_ir:
            continue

        if rotate_flag:
            # Rotate the frame by 180 degree
            adjusted = cv2.rotate(adjusted, cv2.ROTATE_180)

        sb_res = seat_belt_detection_model(adjusted)
        cv2.imshow("IMG", adjusted)
        cv2.waitKey(1)

        df = sb_res.pandas().xyxy[0]
        sb_result = df['name'].tolist()

        if 'with_sb' in sb_result:
            if len(sb_result) >= 1:
                if sb_result[0] == 'with_sb':
                    driver_with_sb_count += 1
            if len(sb_result) >= 2:
                if sb_result[1] == 'with_sb':
                    copassenger_with_sb_count += 1

        total_driver_percentage = (driver_with_sb_count / max(1, driver_with_sb_count)) * 100
        print(f'Driver seatbelt status: {total_driver_percentage:.2f}%')

        if copassenger_with_sb_count != 0:
            total_copassenger_percentage = (copassenger_with_sb_count / max(1, copassenger_with_sb_count)) * 100
            print(f'Co-passenger seatbelt status: {total_copassenger_percentage:.2f}%')

    # Close azure camera
    device.close()


def trigger():
    for i in range(0, 100):
        print('Triggered')

        # Get user input
        user_input = '1'

        # Encode the input as bytes
        message = user_input.encode('utf-8')

        # Send the message to the broadcast address and port
        sock.sendto(message, (broadcast_address, port))
    generate_frames()


def generate_frames():
    while True:
        # Get capture
        capture = device.update()

        ret_ir, ir = capture.get_ir_image()  # IR image

        # call convertScaleAbs function, just for visualisation
        adjusted = cv2.convertScaleAbs(ir, alpha=alpha, beta=beta)

        if not ret_ir:
            continue

        if rotate_flag:
            # Rotate the frame by 180 degree
            adjusted = cv2.rotate(adjusted, cv2.ROTATE_180)

        # Compress the frame using JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Adjust quality as needed
        _, ir_jpeg = cv2.imencode('.jpg', adjusted, encode_param)

        # Send the compressed frames
        socket.send_multipart([b"IR", ir_jpeg])


fc_rec()
sb(60)
time.sleep(1)
trigger()
