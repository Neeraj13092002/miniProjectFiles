import torch
import cv2
import time
import socket as sc
from azure_data_capture import capture_data
import zmq
import threading
import numpy as np

# Set the broadcast address
broadcast_address = '172.1.40.24'
port = 3732

# Set to 1 if rotation by 180 degree is needed.
rotate_flag = 0

# Set this if image display is needed.
display_images_flag = 1

# Initialize ZeroMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")  # Bind to port for broadcasting

alpha = 0.2  # Contrast control
beta = 0.009  # Brightness control

# Create a UDP socket
sock = sc.socket(sc.AF_INET, sc.SOCK_DGRAM)

# Bind the socket to the port
sock.bind(('0.0.0.0', port))

# Enable broadcasting mode
sock.setsockopt(sc.SOL_SOCKET, sc.SO_BROADCAST, 1)

ir = np.empty((512, 512))
rgb = np.empty((720, 1280, 3))


def get_data():
    print("Initializing Azure...")
    global ir, rgb

    while True:
        ir,rgb,ret_rgb = capture_data()
        
        if not ret_rgb:
            continue
        print(ir.shape, rgb.shape)


def face_reco():
    import face_recognition
    print('Scanning faces...')
    while True:
        # ir, _ = capture_data()  # IR data

        # call convertScaleAbs function, just for visualisation
        adjusted = ir

        if rotate_flag:
            # Rotate the frame by 180 degree
            adjusted = cv2.rotate(adjusted, cv2.ROTATE_180)

        # Call the real-time face recognition function from the imported script
        class_labels = face_recognition.real_time_face_recognition(adjusted, display_images_flag)

        if 'Raafay' in class_labels:
            cleaned_name = class_labels[0].replace("[", "").replace("]", "").replace("'", "")
            print(f'Welcome, {cleaned_name}!')
            print()
            break
        if class_labels is None:
            break

    if display_images_flag:
        cv2.destroyAllWindows()

    time.sleep(2)


def seat_belt_detection(time_to_capture):
    seat_belt_detection_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                               path=r'sb_model.pt',
                                               force_reload=False)

    print('Checking Seatbelt Status.')
    print()
    driver_with_sb_count = 0
    copassenger_with_sb_count = 0

    start_time = time.time()
    while (time.time() - start_time) < time_to_capture:
        for i in range(100):
            # ir, _ = capture_data()  # IR image


            # call convertScaleAbs function, just for visualisation
            adjusted = cv2.convertScaleAbs(ir, alpha=alpha, beta=beta)

            if rotate_flag:
                # Rotate the frame by 180 degree
                adjusted = cv2.rotate(adjusted, cv2.ROTATE_180)

            sb_res = seat_belt_detection_model(adjusted)
            sb_detections = sb_res.render()[0]

            if display_images_flag:
                cv2.imshow("IMG", sb_detections)
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
        print()

        if copassenger_with_sb_count != 0:
            total_copassenger_percentage = (copassenger_with_sb_count / max(1, copassenger_with_sb_count)) * 100
            print(f'Co-passenger seatbelt status: {total_copassenger_percentage:.2f}%')
            print()
        break

    if display_images_flag:
        cv2.destroyAllWindows()

    time.sleep(5)


def send_trigger():
    for i in range(0, 100):
        trigger_msg = '1'
        # Encode the input as bytes
        message = trigger_msg.encode('utf-8')
        # Send the message to the broadcast address and port
        sock.sendto(message, (broadcast_address, port))

    print('Triggered DMS on Client PC.')
    print()
    broadcast_frames()
    


def broadcast_frames():
    print('Broadcasting Data.')
    print()
    while True:
        # ir, rgb = capture_data()  # IR image and RGB image

        # call convertScaleAbs function, just for visualisation
        adjusted = cv2.convertScaleAbs(ir, alpha=alpha, beta=beta)

        if rotate_flag:
            # Rotate the frame by 180 degree
            adjusted = cv2.rotate(adjusted, cv2.ROTATE_180)
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)

        if display_images_flag:
            try:          
                cv2.imshow("IR Image", adjusted)
                # # Resize the RGB image
                # width = 960  # Example width
                # height = 540  # Example height
                # resized_rgb = cv2.resize(rgb, (width, height))
                cv2.imshow("RGB Image", rgb)
                cv2.waitKey(1)
            except:
                print("ERROR resizing")
        # Compress the frame using JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Adjust quality as needed
        _, ir_jpeg = cv2.imencode('.jpg', adjusted, encode_param)
        _, rgb_jpeg = cv2.imencode('.jpg', rgb, encode_param)

        # Send the compressed frames
        socket.send_multipart([b"IR", ir_jpeg])
        socket.send_multipart([b"RGB", rgb_jpeg])

broadcast_thread = threading.Thread(target=get_data)
broadcast_thread.start()

# face_reco()
seat_belt_detection(60)
send_trigger()
# get_data()