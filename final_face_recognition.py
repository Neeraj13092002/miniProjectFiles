import torch
import time
import cv2
from datetime import datetime
import pykinect_azure as pykinect
from collections import defaultdict, Counter 

# Set to 1 if rotation by 180 degree is needed.
rotate_flag = 0

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration


device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
# device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_YUY2
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
device = pykinect.start_device(config=device_config)

alpha = 0.2  # Contrast control
beta = 0.009  # Brightness control

fc_flag = 1
def fc_rec(start_time):
    import testing_test_fr
    global fc_flag
    while True:
        # Get capture
        capture = device.update()
        ret_ir, ir = capture.get_ir_image()  # IR image

        if not ret_ir:
            continue

        # call convertScaleAbs function, just for visualisation
        adjusted = ir

        if rotate_flag:
            # Rotate the frame by 180 degree
            adjusted = cv2.rotate(adjusted, cv2.ROTATE_180)
        # Call the real-time face recognition function from the imported script
        print("start:",start_time)
        detection_counts = defaultdict(int)
        class_labels = testing_test_fr.real_time_face_recognition(adjusted)
        
        for label in class_labels:
            detection_counts[label] += 1

        #The Most Common Label is flagged as Unknown at the start
        most_common_label = "UNKNOWN"
        #Duration is currently set to 1 Minute
        duration = 50
            # Check if the time window has passed
        current_time = datetime.now()
        print("Current:",current_time)
        print("Diff:",(current_time-start_time).total_seconds())
        diff = (current_time-start_time).total_seconds()
        print(class_labels)
        if (diff >= duration):
                # Determine the most frequent label within the time window
            if detection_counts:
                most_common_label, _ = Counter(detection_counts).most_common(1)[0]
                print(f"Most common label in the last {duration} seconds: {most_common_label}")
                break

                # Reset the detection counts and start time
            # detection_counts.clear()
            # start_time = current_time

    fc_flag = 1
    # cv2.destroyAllWindows()
    time.sleep(5)

start_time = datetime.now()
fc_rec(start_time)