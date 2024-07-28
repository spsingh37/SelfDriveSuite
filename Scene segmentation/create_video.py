import cv2
import os
import re
import argparse

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def create_video_from_images(image_folder, output_video, frame_rate=30, width=None, height=None):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted_alphanumeric(images)

    # Filter images that match the naming pattern "frame<number>"
    images = [img for img in images if re.match(r'frame\d+\.png', img)]

    if not images:
        print("No images found in the directory that match the pattern 'frame<number>.png'")
        return

    # Read the first image to get the dimensions if width and height are not specified
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if width is None or height is None:
        height, width, layers = frame.shape

    # Initialize video writer
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        resized_frame = cv2.resize(frame, (width, height))
        video.write(resized_frame)

    video.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine images into a video.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing the images.")
    parser.add_argument("output_video", type=str, help="Path to save the output video file.")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate for the video. Default is 30.")
    parser.add_argument("--width", type=int, help="Width of the output video.")
    parser.add_argument("--height", type=int, help="Height of the output video.")

    args = parser.parse_args()
    create_video_from_images(args.image_folder, args.output_video, args.frame_rate, args.width, args.height)
