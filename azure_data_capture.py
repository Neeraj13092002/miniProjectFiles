import pykinect_azure as pykinect
import cv2
import zmq

# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_YUY2
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15

# Start device
device = pykinect.start_device(config=device_config)


def capture_data():
    # Get capture   
    capture = device.update()
    ret_ir, ir_frame = capture.get_ir_image()  # IR image
    ret_rgb, rgb_frame = capture.get_color_image()  # RGB image

    return ir_frame, rgb_frame, ret_rgb
capture_data()