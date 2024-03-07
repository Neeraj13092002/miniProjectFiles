import cv2 as cv
import os


def capture_images(output_folder, num_images=10):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the default camera (0) or you can specify the camera by passing its index
    cap = cv.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    print("Press 'Space' to capture an image. Press 'Esc' to exit.")

    image_count = 0
    while image_count < num_images:
        ret, frame = cap.read()  # Read a frame from the camera

        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        # Display the frame
        cv.imshow('Press Space to Capture', frame)

        # Wait for key press
        key = cv.waitKey(1)

        if key == 27:  # If 'Esc' key is pressed, exit the loop
            break
        elif key == 32:  # If 'Space' key is pressed, save the image
            image_count += 1
            image_name = f"image_{image_count}.jpg"
            image_path = os.path.join(output_folder, image_name)
            cv.imwrite(image_path, frame)
            print(f"Image {image_name} saved.")

    # Release the camera and close OpenCV windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    output_folder = "captured_images"
    num_images = 10  # Number of images to capture
    capture_images(output_folder, num_images)
