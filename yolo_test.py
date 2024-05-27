import cv2
import zmq
import numpy as np
import torch


# Load YOLOv5 models
face_recognition_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'D:\dms_ir\dms_ir_TL\weights\best.pt', force_reload=True)
seat_belt_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'D:\seat_belt\ir_only_sb_model\exp12\weights\best.pt', force_reload=True)

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://172.1.40.24:5555")  # Connect to the sender's address

# Subscribe to all messages
socket.setsockopt_string(zmq.SUBSCRIBE, "")

while True:
    # Receive the frame type
    frame_type, frame_bytes = socket.recv_multipart()

    # Convert bytes to numpy array
    nparr = np.frombuffer(frame_bytes, dtype=np.uint8)

    # Decode the JPEG bytes into a frame
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Display the frame based on its type
    if frame_type == b"IR":
        dms_res = face_recognition_model(frame)
        print('DMS:', dms_res)
        cv2.imshow("IR Frame", frame)
    elif frame_type == b"RGB":
        sb_res = seat_belt_detection_model(frame)
        print('Seat Belt', sb_res)
        cv2.imshow("RGB Frame", frame)

    cv2.waitKey(1)
