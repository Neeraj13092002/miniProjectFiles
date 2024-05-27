import cv2
import zmq
import numpy as np
import time

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://172.1.40.24:5555")  # Connect to the sender's address

# Subscribe to all messages
socket.setsockopt_string(zmq.SUBSCRIBE, "")

# Variables to track FPS
prev_time_rgb = time.time()
prev_time_ir = time.time()
frame_count_rgb = 0
frame_count_ir = 0

while True:
    # Receive the frame type
    frame_type, frame_bytes = socket.recv_multipart()

    # Convert bytes to numpy array
    nparr = np.frombuffer(frame_bytes, dtype=np.uint8)

    # Decode the JPEG bytes into a frame
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Display the frame based on its type
    if frame_type == b"IR":
        cv2.imshow("IR Frame", frame)
        frame_count_ir += 1
        curr_time_ir = time.time()
        if curr_time_ir - prev_time_ir >= 1:
            fps_ir = frame_count_ir / (curr_time_ir - prev_time_ir)
            print("IR FPS:", fps_ir)
            frame_count_ir = 0
            prev_time_ir = curr_time_ir
    elif frame_type == b"RGB":
        cv2.imshow("RGB Frame", frame)
        frame_count_rgb += 1
        curr_time_rgb = time.time()
        if curr_time_rgb - prev_time_rgb >= 1:
            fps_rgb = frame_count_rgb / (curr_time_rgb - prev_time_rgb)
            print("RGB FPS:", fps_rgb)
            frame_count_rgb = 0
            prev_time_rgb = curr_time_rgb

    cv2.waitKey(1)
